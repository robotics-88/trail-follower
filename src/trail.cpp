/*
© 2024 Robotics 88
Author: Erin Linebarger <erin@robotics88.com>
*/

#include "trail_follower/trail.h"
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_stick.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/progressive_morphological_filter.h>

#include <cmath>
#include <thread>

using std::placeholders::_1;

namespace trail_follower {

Trail::Trail()
    : Node("trail_follower"),
      is_active_(false),
      planning_horizon_(10.0),
      trail_goal_enabled_(false),
      pub_rate_(2.0),
      point_cloud_topic_("/cloud_to_use"),
      segment_distance_threshold_(0.01),
      pmf_max_window_size_(10),
      pmf_slope_(1.0),
      pmf_initial_distance_(0.5),
      pmf_max_distance_(3.0),
      last_pub_time_(0, 0, RCL_ROS_TIME),
      cloud_init_(false),
      has_first_trailpt_(false) {
    // Get params
    this->declare_parameter("pub_rate", pub_rate_);
    this->declare_parameter("point_cloud_topic", point_cloud_topic_);
    this->declare_parameter("segment_distance_threshold", segment_distance_threshold_);
    this->declare_parameter("pmf_max_window_size", pmf_max_window_size_);
    this->declare_parameter("pmf_slope", pmf_slope_);
    this->declare_parameter("pmf_initial_distance", pmf_initial_distance_);
    this->declare_parameter("pmf_max_distance", pmf_max_distance_);

    this->get_parameter("pub_rate", pub_rate_);
    this->get_parameter("point_cloud_topic", point_cloud_topic_);
    this->get_parameter("segment_distance_threshold", segment_distance_threshold_);
    this->get_parameter("pmf_max_window_size", pmf_max_window_size_);
    this->get_parameter("pmf_slope", pmf_slope_);
    this->get_parameter("pmf_initial_distance", pmf_initial_distance_);
    this->get_parameter("pmf_max_distance", pmf_max_distance_);

    // Set up pubs and subs
    mavros_local_pos_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/mavros/vision_pose/pose", rclcpp::SensorDataQoS(),
        std::bind(&Trail::localPositionCallback, this, _1));

    point_cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        point_cloud_topic_, 10, std::bind(&Trail::pointCloudCallback, this, _1));

    cloud_ground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_ground", 10);
    cloud_nonground_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_nonground", 10);
    cloud_cluster_pub_ =
        this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_clusters", 10);

    trail_line_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/trail_line", 10);
    trail_marker_.scale.x = 3.0;
    trail_marker_.type = visualization_msgs::msg::Marker::LINE_STRIP;
    trail_marker_.action = visualization_msgs::msg::Marker::ADD;
    trail_marker_.id = 0;
    std_msgs::msg::ColorRGBA yellow;
    yellow.r = 1.0;
    yellow.g = 1.0;
    yellow.b = 0;
    yellow.a = 1.0;
    trail_marker_.color = yellow;
    trail_marker_.lifetime = rclcpp::Duration(0.0, 0.0);
    trail_ends_pub_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>("/trail_ends", 10);

    trail_goal_pub_ =
        this->create_publisher<geometry_msgs::msg::PoseStamped>("/explorable_goal", 10);

    // TODO remove service, replace with pub/sub or param cb
    trail_enabled_service_ = this->create_service<rcl_interfaces::srv::SetParametersAtomically>(
        "trail_enabled_service", std::bind(&Trail::setTrailsEnabled, this, std::placeholders::_1,
                                           std::placeholders::_2, std::placeholders::_3));

    param_subscriber_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    startParamMonitoring(); // Use timer to wait for task_manager to load perception registry
}

Trail::~Trail() {}

void Trail::localPositionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    current_pose_ = *msg;
}

void Trail::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!is_active_) {
        return;
    }
    // Convert ROS msg to PCL and store
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud);
    std_msgs::msg::Header header = msg->header;

    // Only run processing at limited rate
    rclcpp::Duration dur = this->get_clock()->now() - last_pub_time_;
    if (dur.seconds() < (1.0 / pub_rate_)) {
        return;
    } else {
        doGroundAndTrail(cloud, header);
        last_pub_time_ = this->get_clock()->now();
    }
}

void Trail::parameterCallback(const rclcpp::Parameter &param) {
    is_active_ = param.as_bool();
    RCLCPP_INFO(this->get_logger(), "Trail follower node active: %s", is_active_ ? "true" : "false");
}

void Trail::startParamMonitoring() {
    param_monitor_timer_ = this->create_wall_timer(
        std::chrono::seconds(1),
        [this]() {
            static bool callback_registered = false;

            if (!callback_registered) {
                try {
                    cb_handle_ = param_subscriber_->add_parameter_callback(
                        "/task_manager/trail_follower/set_node_active",
                        std::bind(&Trail::parameterCallback, this, std::placeholders::_1),
                        "task_manager/task_manager"
                    );
                    RCLCPP_INFO(this->get_logger(), "✅ Parameter callback registered for task_manager:trail_follower/set_node_active");
                    callback_registered = true;
                    param_monitor_timer_->cancel();  // stop retrying
                } catch (const std::exception &e) {
                    RCLCPP_WARN(this->get_logger(), "Waiting for task_manager param to become available: %s", e.what());
                }
            }
        });
}


void Trail::doGroundAndTrail(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                             const std_msgs::msg::Header header) {
    // Extract ground returns
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nonground(new pcl::PointCloud<pcl::PointXYZ>());
    pmf_ground_extraction(cloud, cloud_ground, cloud_nonground);

    // Cluster ground returns to find the trail
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clustered(new pcl::PointCloud<pcl::PointXYZ>());
    findTrail(cloud_ground, header, cloud_clustered);

    // Convert to ROS msg and publish
    sensor_msgs::msg::PointCloud2 cloud_ground_msg, cloud_nonground_msg, cloud_clustered_msg;
    pcl::toROSMsg(*cloud_ground, cloud_ground_msg);
    pcl::toROSMsg(*cloud_nonground, cloud_nonground_msg);
    pcl::toROSMsg(*cloud_clustered, cloud_clustered_msg);
    cloud_ground_pub_->publish(cloud_ground_msg);
    cloud_nonground_pub_->publish(cloud_nonground_msg);
    cloud_clustered_msg.header = header;
    cloud_cluster_pub_->publish(cloud_clustered_msg);

    trail_marker_.header = header;
    trail_line_pub_->publish(trail_marker_);
}

void Trail::findTrail(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                      const std_msgs::msg::Header header,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clustered) {
    // Set up the clustering
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(1.0); // Adjust tolerance as necessary
    ec.setMinClusterSize(50);    // Minimum number of points to form a cluster
    // ec.setMaxClusterSize(25000); // Maximum number of points to form a cluster
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);
    if (cluster_indices.size() == 0)
        return;
    // TEST: Get largest cluster
    int max_index = 0, max_size = 0;
    for (int ii = 0; ii < cluster_indices.size(); ii++) {
        if (cluster_indices.at(ii).indices.size() > max_size) {
            max_index = ii;
            max_size = cluster_indices.at(ii).indices.size();
        }
    }

    // pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndices::Ptr cluster(new pcl::PointIndices());
    *cluster = cluster_indices.at(max_index);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(cluster);
    extract.filter(*cloud_clustered);

    // Extract line segment and append to trail marker list
    extractLineSegment(cloud_clustered, header);
}

void Trail::extractLineSegment(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clustered,
                               const std_msgs::msg::Header header) {
    // Perform RANSAC plane fitting
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.1); // Set distance threshold for inliers
    seg.setInputCloud(cloud_clustered);
    seg.segment(*inliers, *coefficients);
    // TODO add metric checking for percentage of inliers to determine if no trail found

    if (inliers->indices.size() == 0) {
        RCLCPP_INFO(this->get_logger(), "Could not estimate a linear model for the given dataset.");
    }

    // Extract the line segment endpoints from the coefficients
    double factor = 5.0;
    geometry_msgs::msg::Point point1, point2;
    point1.x = coefficients->values[0] - factor * coefficients->values[3];
    point1.y = coefficients->values[1] - factor * coefficients->values[4];
    point1.z = coefficients->values[2] - factor * coefficients->values[5];
    trail_marker_.points.push_back(point1);

    // Second set of coefficients is a direction vector
    point2.x = coefficients->values[0] + factor * coefficients->values[3];
    point2.y = coefficients->values[1] + factor * coefficients->values[4];
    point2.z = coefficients->values[2] + factor * coefficients->values[5];
    trail_marker_.points.push_back(point2);

    // Viz (add arg)
    visualization_msgs::msg::MarkerArray markers_msg;
    visualization_msgs::msg::Marker m;
    std_msgs::msg::ColorRGBA yellow;
    yellow.r = 1.0;
    yellow.g = 1.0;
    yellow.b = 0;
    yellow.a = 1.0;
    m.header = header;
    double scale = 1.0;
    m.scale.x = scale;
    m.scale.y = scale;
    m.scale.z = scale;
    m.type = visualization_msgs::msg::Marker::SPHERE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.color = yellow;
    m.ns = "trail";
    m.id = 1;
    m.pose.position = point1;
    markers_msg.markers.push_back(m);
    m.id = 2;
    m.pose.position = point2;
    markers_msg.markers.push_back(m);
    trail_ends_pub_->publish(markers_msg);

    if (trail_goal_enabled_) {
        geometry_msgs::msg::Point send_point;
        if (has_first_trailpt_) {
            // Which point minimizes change in heading? TODO, test edge cases, eg what about
            // switchbacks?
            tf2::Quaternion quat_tf;
            tf2::convert(current_pose_.pose.orientation, quat_tf);
            tf2::Matrix3x3 m(quat_tf);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw); // yaw already in ENU frame radians
            double theta1, theta2;
            findAngle(current_pose_.pose.position.x, current_pose_.pose.position.y, yaw, point1.x,
                      point1.y, theta1);
            findAngle(current_pose_.pose.position.x, current_pose_.pose.position.y, yaw, point2.x,
                      point2.y, theta2);
            if (theta1 < theta2) {
                send_point = point1;
            } else {
                send_point = point2;
            }
        } else {
            // Send the farther point
            double d1 = sqrt(pow(last_trail_point_.pose.position.x - point1.x, 2) +
                             pow(last_trail_point_.pose.position.y - point1.y, 2));
            double d2 = sqrt(pow(last_trail_point_.pose.position.x - point2.x, 2) +
                             pow(last_trail_point_.pose.position.y - point2.y, 2));
            if (d1 < d2) {
                send_point = point2;
            } else {
                send_point = point1;
            }
            has_first_trailpt_ = true;
        }
        geometry_msgs::msg::PoseStamped point_msg;
        point_msg.header = header;
        point_msg.pose.position = send_point;
        trail_goal_pub_->publish(point_msg);
        last_trail_point_ = point_msg;
    }
}

void Trail::findAngle(const double x1, const double y1, const double theta1, const double x2,
                      const double y2, double &theta_out) {
    // Line ax + by + c = 0, given by x1, y1, theta1
    double a = -1 * tan(theta1);
    double b = 1;
    double c = -y1 + tan(theta1) * x1;

    // Get projected point, (x2, y2) onto line
    double temp = -1 * (a * x2 + b * y2 + c) / (a * a + b * b);
    double x = temp * a + x2;
    double y = temp * b + y2;

    // Compute angle of projection
    double hyp = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
    double adj = sqrt(pow(x1 - x, 2) + pow(y1 - y, 2));
    theta_out = acos(adj / hyp);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
Trail::findMaximumPlanar(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clustered,
                         const std::vector<pcl::PointIndices> cluster_indices) {
    // Iterate through each cluster
    int index = 0, count = 0;
    float max_ratio = 0.0;
    for (int ii = 0; ii < cluster_indices.size(); ii++) {
        // for (pcl::PointIndices cluster : cluster_indices) {
        if (cluster_indices.at(ii).indices.size() < 1000) {
            continue;
        }
        pcl::PointIndices::Ptr cluster(new pcl::PointIndices());
        *cluster = cluster_indices.at(ii);
        pcl::PointCloud<pcl::PointXYZ> cluster_points;
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_clustered);
        extract.setIndices(cluster);
        extract.filter(cluster_points);

        // Perform RANSAC plane fitting
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_LINE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.1); // Set distance threshold for inliers
        seg.setInputCloud(cluster_points.makeShared());
        seg.segment(*inliers, *coefficients);

        // Calculate planarity metric (e.g., inlier ratio)
        float inlier_ratio = static_cast<float>(inliers->indices.size()) / cluster_points.size();

        // Store the planarity metric for comparison
        if (inlier_ratio > max_ratio) {
            index = count;
        }
        count++;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndices::Ptr cluster(new pcl::PointIndices());
    *cluster = cluster_indices.at(index);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_clustered);
    extract.setIndices(cluster);
    extract.filter(*cluster_points);

    return cluster_points;
}

void Trail::segment_plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nonplane) {

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;

    // Set up segmentation
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(segment_distance_threshold_);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0) {
        RCLCPP_INFO(this->get_logger(), "Could not estimate a planar model from pointcloud.");
        return;
    }

    // Extract segment from cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_plane);

    // Extract opposite segment
    extract.setNegative(true);
    extract.filter(*cloud_nonplane);
}

void Trail::segment_cylinders(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cylinders,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_noncylinders) {}

void Trail::pmf_ground_extraction(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nonground) {

    pcl::PointIndicesPtr ground(new pcl::PointIndices);

    // Create the filtering object
    pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
    pmf.setInputCloud(cloud);
    pmf.setMaxWindowSize(pmf_max_window_size_);
    pmf.setSlope(pmf_slope_);
    pmf.setInitialDistance(pmf_initial_distance_);
    pmf.setMaxDistance(pmf_max_distance_);

    pmf.extract(ground->indices);

    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(ground);
    extract.filter(*cloud_ground);

    // Extract non-ground returns
    extract.setNegative(true);
    extract.filter(*cloud_nonground);
}

void Trail::voxel_grid_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size) {
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    sor.filter(*cloud);
}

float Trail::get_percent_above(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    if (cloud->points.size() < 1) {
        RCLCPP_ERROR(this->get_logger(), "Attempting to process invalid PCL");
        return -1.0;
    }

    int num_pts_above = 0;
    for (const auto &point : cloud->points) {
        if (point.z > current_pose_.pose.position.z) {
            num_pts_above++;
        }
    }

    return static_cast<float>(num_pts_above) / cloud->points.size();
}

bool Trail::setTrailsEnabled(
    const std::shared_ptr<rmw_request_id_t> /*request_header*/,
    const std::shared_ptr<rcl_interfaces::srv::SetParametersAtomically::Request> req,
    const std::shared_ptr<rcl_interfaces::srv::SetParametersAtomically::Response> resp) {
    for (int ii = 0; ii < req->parameters.size(); ii++) {
        if (req->parameters.at(ii).name == "trails_enabled") {
            trail_goal_enabled_ = req->parameters.at(ii).value.bool_value;
        }
    }
    auto result = rcl_interfaces::msg::SetParametersResult();
    result.successful = true;
    resp->result = result;
}

} // namespace trail