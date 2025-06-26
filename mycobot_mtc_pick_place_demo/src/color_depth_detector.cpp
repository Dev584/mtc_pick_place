#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <mycobot_interfaces/msg/detected_object.hpp>

#include <vector>

struct Detection
{
  std::string   id;
  cv::Rect      box;
  cv::Point     centroid;
};

class ColorDepthDetector : public rclcpp::Node {
public:
  ColorDepthDetector();

private:
  // Subscribers & publishers
  image_transport::Subscriber                                          image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr             depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr        info_sub_;
  rclcpp::Publisher<mycobot_interfaces::msg::DetectedObject>::SharedPtr det_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr          pose_pub_;
  image_transport::Publisher                                          debug_pub_;

  // Camera intrinsics
  bool   has_info_{false};
  float  fx_{0.0f}, fy_{0.0f}, cx_{0.0f}, cy_{0.0f};

  // Detected blobs
  std::vector<Detection> detections_;

  // Callbacks
  void onInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void onColor(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  void onDepth(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
};

ColorDepthDetector::ColorDepthDetector()
: Node("color_depth_detector")
{
  // Camera info for intrinsics
  info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera_head/depth/camera_info", 10,
    std::bind(&ColorDepthDetector::onInfo, this, std::placeholders::_1));

  // Color image
  image_transport::ImageTransport it(this->shared_from_this());
  image_sub_ = it.subscribe(
    "/camera_head/color/image_raw", 10,
    std::bind(&ColorDepthDetector::onColor, this, std::placeholders::_1));

  // Depth image
  depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera_head/depth/image_rect_raw", 10,
    std::bind(&ColorDepthDetector::onDepth, this, std::placeholders::_1));

  // Publishers
  det_pub_  = this->create_publisher<mycobot_interfaces::msg::DetectedObject>(
    "/detected_objects", 10);
  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
    "/detected_cylinders", 10);
  debug_pub_ = it.advertise("/debug_image", 10);
}

void ColorDepthDetector::onInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  fx_ = msg->k[0];  fy_ = msg->k[4];
  cx_ = msg->k[2];  cy_ = msg->k[5];
  has_info_ = true;
}

void ColorDepthDetector::onColor(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  // convert color image to OpenCV Mat
  if (!has_info_) return;
  cv::Mat bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
  cv::Mat blur;
  cv::GaussianBlur(bgr, blur, cv::Size(5, 5), 0);
  cv::Mat hsv;
  cv::cvtColor(blur, hsv, cv::COLOR_BGR2HSV);

  // prepare detection list
  std::vector<Detection> new_detections;

  // define HSV ranges for colors
  struct ColorRange { std::string id; cv::Scalar lower; cv::Scalar upper; };
  std::vector<ColorRange> ranges = {
    {"red",    cv::Scalar(0, 120, 70),   cv::Scalar(10, 255, 255)},
    {"red",    cv::Scalar(170, 120, 70), cv::Scalar(180, 255, 255)},
    {"green",  cv::Scalar(35,  100, 100), cv::Scalar(85, 255, 255)},
    {"yellow", cv::Scalar(20,  100, 100), cv::Scalar(30, 255, 255)}
  };

  for (const auto &range : ranges) {
    cv::Mat mask;
    cv::inRange(hsv, range.lower, range.upper, mask);
    // morphological cleanup
    cv::erode(mask, mask, cv::Mat(), cv::Point(-1,-1), 2);
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1,-1), 2);

    // find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) continue;

    // find largest contour
    auto max_it = std::max_element(contours.begin(), contours.end(),
      [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b){
        return cv::contourArea(a) < cv::contourArea(b);
      });
    auto &cnt = *max_it;
    cv::Rect box = cv::boundingRect(cnt);
    cv::Point centroid(box.x + box.width/2, box.y + box.height/2);

    // record detection
    Detection d;
    d.id = range.id;
    d.box = box;
    d.centroid = centroid;
    new_detections.push_back(d);
  }

  // update detections_
  detections_ = std::move(new_detections);
}

void ColorDepthDetector::onDepth(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  if (!has_info_ || detections_.empty())
    return;

  // Convert depth image
  cv::Mat depth = cv_bridge::toCvCopy(msg)->image;

  // Only use the first detection
  const auto &d = detections_.front();
  float z = depth.at<float>(d.centroid.y, d.centroid.x);
  float x = (d.centroid.x - cx_) * z / fx_;
  float y = (d.centroid.y - cy_) * z / fy_;

  // Publish DetectedObject
  mycobot_interfaces::msg::DetectedObject out;
  out.id         = d.id;
  out.xmin       = d.box.x;
  out.ymin       = d.box.y;
  out.width      = d.box.width;
  out.height     = d.box.height;
  out.position.x = x;
  out.position.y = y;
  out.position.z = z;
  det_pub_->publish(out);

  // Publish PoseArray with a single pose
  geometry_msgs::msg::PoseArray pose_array;
  pose_array.header.stamp = this->now();
  pose_array.header.frame_id = "camera_link";
  geometry_msgs::msg::Pose p;
  p.position.x = x;
  p.position.y = y;
  p.position.z = z;
  p.orientation.w = 1.0;
  pose_array.poses.push_back(p);
  pose_pub_->publish(pose_array);

  detections_.clear();
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ColorDepthDetector>());
  rclcpp::shutdown();
  return 0;
}
