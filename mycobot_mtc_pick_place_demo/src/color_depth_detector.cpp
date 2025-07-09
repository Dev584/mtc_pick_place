#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <mycobot_interfaces/msg/detected_object.hpp>

#include <vector>
#include <memory>

struct Detection
{
  std::string id;
  cv::Rect    box;
  cv::Point   centroid;
};

class ColorDepthDetector : public rclcpp::Node
{
public:
  ColorDepthDetector();

private:
  // Subscribers & publishers
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr        info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr             depth_sub_;
  image_transport::Subscriber                                          color_sub_;
  rclcpp::Publisher<mycobot_interfaces::msg::DetectedObject>::SharedPtr det_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr          pose_pub_;
  image_transport::Publisher                                          debug_pub_;

  // Transport (initialized in onInfo)
  std::shared_ptr<image_transport::ImageTransport>                    it_;

  // Intrinsics
  bool   has_info_{false};
  float  fx_{0}, fy_{0}, cx_{0}, cy_{0};

  // Latest color detections
  std::vector<Detection> detections_;

  // Callbacks
  void onInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void onColor(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  void onDepth(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
};

ColorDepthDetector::ColorDepthDetector()
: Node("color_depth_detector")
{
  // 1) Camera info subscription
  info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera_head/depth/camera_info", 10,
    std::bind(&ColorDepthDetector::onInfo, this, std::placeholders::_1)
  );

  // 2) Depth image subscription
  depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera_head/depth/image_rect_raw", 10,
    std::bind(&ColorDepthDetector::onDepth, this, std::placeholders::_1)
  );

  // 3) Publishers for detected objects and poses
  det_pub_  = this->create_publisher<mycobot_interfaces::msg::DetectedObject>(
    "/detected_objects", 10
  );
  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
    "/detected_cylinders", 10
  );
}

void ColorDepthDetector::onInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  // Cache intrinsics once
  fx_ = msg->k[0];  fy_ = msg->k[4];
  cx_ = msg->k[2];  cy_ = msg->k[5];
  has_info_ = true;

  // Only build ImageTransport _after_ node is fully constructed
  if (!it_) {
    auto base = rclcpp::Node::shared_from_this();
    auto self = std::static_pointer_cast<ColorDepthDetector>(base);

    it_ = std::make_shared<image_transport::ImageTransport>(self);

    color_sub_ = it_->subscribe(
      "/camera_head/color/image_raw", 10,
      std::bind(&ColorDepthDetector::onColor, this, std::placeholders::_1)
    );
    debug_pub_ = it_->advertise("/debug_image", 10);

    RCLCPP_INFO(get_logger(), "ImageTransport and color subscription initialized");
  }
}

void ColorDepthDetector::onColor(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  RCLCPP_DEBUG(get_logger(), "onColor()");
  if (!has_info_) {
    RCLCPP_WARN(get_logger(), "Skipping color until camera info arrives");
    return;
  }

  // Convert to BGR and blur & HSV
  auto bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
  cv::Mat blur, hsv;
  cv::GaussianBlur(bgr, blur, cv::Size(5,5), 0);
  cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);

  // Define color ranges
  struct ColorRange { std::string id; cv::Scalar lo, hi; };
  std::vector<ColorRange> ranges = {
    {"red",    {  0,120, 70}, { 10,255,255}},
    {"red",    {170,120, 70}, {180,255,255}},
    {"green",  { 35,100,100}, { 85,255,255}},
    {"yellow", { 20,100,100}, { 30,255,255}}
  };

  std::vector<Detection> new_dets;
  for (auto &r : ranges) {
    cv::Mat mask;
    cv::inRange(hsv, r.lo, r.hi, mask);
    cv::erode(mask, mask, {}, cv::Point(-1,-1), 2);
    cv::dilate(mask, mask, {}, cv::Point(-1,-1), 2);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) continue;

    // Largest contour
    auto best = *std::max_element(contours.begin(), contours.end(),
      [](auto &a, auto &b){ return cv::contourArea(a) < cv::contourArea(b); });
    cv::Rect box = cv::boundingRect(best);
    cv::Point c{ box.x + box.width/2, box.y + box.height/2 };

    new_dets.push_back({ r.id, box, c });
  }

  detections_ = std::move(new_dets);
  RCLCPP_INFO(get_logger(), "Detected %zu color blobs", detections_.size());
}

void ColorDepthDetector::onDepth(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  RCLCPP_DEBUG(get_logger(), "onDepth()");
  if (!has_info_) {
    RCLCPP_WARN(get_logger(), "Skipping depth until camera info arrives");
    return;
  }
  if (detections_.empty()) {
    RCLCPP_WARN(get_logger(), "Skipping depth: no color detections");
    return;
  }

  // Convert depth image
  cv::Mat depth = cv_bridge::toCvCopy(msg)->image;

  geometry_msgs::msg::PoseArray pa;
  pa.header = msg->header;
  pa.header.frame_id = "camera_head_link";

  for (auto &d : detections_) {
    float z = depth.at<float>(d.centroid.y, d.centroid.x);
    float x = (d.centroid.x - cx_) * z / fx_;
    float y = (d.centroid.y - cy_) * z / fy_;

    // Publish per-object message
    mycobot_interfaces::msg::DetectedObject obj;
    obj.id     = d.id;
    obj.xmin   = d.box.x;
    obj.ymin   = d.box.y;
    obj.width  = d.box.width;
    obj.height = d.box.height;
    obj.position.x = x;
    obj.position.y = y;
    obj.position.z = z;
    det_pub_->publish(obj);

    // Add to PoseArray
    geometry_msgs::msg::Pose p;
    p.position.x = x;  p.position.y = y;  p.position.z = z;
    p.orientation.w = 1.0;
    pa.poses.push_back(p);
  }

  pose_pub_->publish(pa);
  detections_.clear();
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ColorDepthDetector>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
