/*
© 2024 Robotics 88
Author: Erin Linebarger <erin@robotics88.com>
*/

#ifndef TRAIL_FOLLOWER_H_
#define TRAIL_FOLLOWER_H_

#include <rclcpp/rclcpp.hpp>

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/float32.hpp"
#include <pcl/point_cloud.h>

#include "visualization_msgs/msg/marker.hpp"
#include <visualization_msgs/msg/marker_array.hpp>

namespace trail_follower {

/**
 * @class Trail
 * @brief A class for analyzing and segmenting point clouds
 */
class Trail : public rclcpp::Node {

  public:
    Trail();
    ~Trail();

    void localPositionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  private:
    std::string point_cloud_topic_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mavros_local_pos_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_subscriber_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_ground_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_nonground_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_cluster_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr trail_line_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr trail_ends_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr trail_goal_pub_;

    rclcpp::Time last_pub_time_;

    // Main input pointcloud holder
    bool cloud_init_;
    bool has_first_trailpt_;
    geometry_msgs::msg::PoseStamped last_trail_point_;
    double planning_horizon_;

    // Trail line
    visualization_msgs::msg::Marker trail_marker_;
    bool trail_goal_enabled_;
    rclcpp::Service<rcl_interfaces::srv::SetParametersAtomically>::SharedPtr trail_enabled_service_;

    // Params
    double pub_rate_;
    double segment_distance_threshold_;
    int pmf_max_window_size_;
    float pmf_slope_;
    float pmf_initial_distance_;
    float pmf_max_distance_;

    geometry_msgs::msg::PoseStamped current_pose_;

    // Parameter handling for toggle on/off
    bool is_active_;
    std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> cb_handle_;
    rclcpp::TimerBase::SharedPtr param_monitor_timer_;
    void parameterCallback(const rclcpp::Parameter &param);
    void startParamMonitoring();
    // End parameter handling

    void doGroundAndTrail(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                          const std_msgs::msg::Header header);

    void findAngle(const double x1, const double y1, const double theta1, const double x2,
                   const double y2, double &theta_out);
    void findTrail(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                   const std_msgs::msg::Header header,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clustered);
    pcl::PointCloud<pcl::PointXYZ>::Ptr
    findMaximumPlanar(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clustered,
                      const std::vector<pcl::PointIndices> cluster_indices);
    void extractLineSegment(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clustered,
                            const std_msgs::msg::Header header);

    // Segments out a plane from a pointcloud including points within segment_distance_threshold_ of
    // plane model
    void segment_plane(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nonplane);

    void segment_cylinders(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cylinders,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_noncylinders);

    // Runs a PMF filter to extract ground points based on pmf_* params
    void pmf_ground_extraction(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nonground);

    // Runs a voxel grid filter to downsample pointcloud into 3D grid of leaf_size
    void voxel_grid_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size);

    // Determines percentage of pointcloud points above drone
    float get_percent_above(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    bool setTrailsEnabled(
        const std::shared_ptr<rmw_request_id_t> /*request_header*/,
        const std::shared_ptr<rcl_interfaces::srv::SetParametersAtomically::Request> req,
        const std::shared_ptr<rcl_interfaces::srv::SetParametersAtomically::Response> resp);

}; // class Trail

} // namespace trail

#endif