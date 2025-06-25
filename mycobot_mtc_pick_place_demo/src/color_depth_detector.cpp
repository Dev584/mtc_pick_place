#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <mycobot_interfaces/msg/detected_object.hpp>
#include <opencv2/opencv.hpp>

struct Detection {
  cv::Rect box;
  cv::Point centroid;
  std::string id;
};

class ColorDepthDetector : public rclcpp::Node {
public:
  ColorDepthDetector()
  : Node("color_depth_detector")
  {
    auto qos = rclcpp::SensorDataQoS();
    // Subscribers
    image_sub_ = image_transport::create_subscription(
      this, "/camera_head/color/image_raw",
      std::bind(&ColorDepthDetector::onColor, this, std::placeholders::_1),
      "raw", qos.get_rmw_qos_profile());
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/camera_head/depth/image_rect_raw", qos,
      std::bind(&ColorDepthDetector::onDepth, this, std::placeholders::_1));
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera_head/depth/camera_info", qos,
      std::bind(&ColorDepthDetector::onInfo, this, std::placeholders::_1));

    // Publishers
    det_pub_ = create_publisher<mycobot_interfaces::msg::DetectedObject>(
      "detected_objects", 10);
    debug_pub_ = image_transport::create_publisher(this, "debug_image");

    has_info_ = false;
  }

private:
  void onInfo(const sensor_msgs::msg::CameraInfo::SharedPtr m) {
    fx_ = m->k[0]; fy_ = m->k[4];
    cx_ = m->k[2]; cy_ = m->k[5];
    has_info_ = true;
  }

  void onColor(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (!has_info_) return;

    cv::Mat bgr = cv_bridge::toCvShare(msg, "bgr8")->image;
    cv::Mat hsv; cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    detections_.clear();
    // Define color thresholds
    struct Thresh { std::string id; cv::Scalar lo, hi; };
    std::vector<Thresh> thr = {
      // red: two ranges
      {"red",    {  0,100,100}, { 10,255,255}},
      {"red",    {160,100,100}, {180,255,255}},
      // green
      {"green",  { 50,100,100}, { 70,255,255}},
      // yellow
      {"yellow", { 20,150,150}, { 30,255,255}},
    };

    for (auto &t : thr) {
      cv::Mat mask;
      cv::inRange(hsv, t.lo, t.hi, mask);
      // clean up
      cv::erode(mask, mask, {}, cv::Point(-1,-1), 2);
      cv::dilate(mask, mask, {}, cv::Point(-1,-1), 2);

      std::vector<std::vector<cv::Point>> cnts;
      cv::findContours(mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      for (auto &c : cnts) {
        cv::Rect box = cv::boundingRect(c);
        if (box.area() < 500) continue;  // filter noise
        cv::Point cent(box.x + box.width/2, box.y + box.height/2);
        detections_.push_back({box, cent, t.id});
        // Draw for debug
        cv::rectangle(bgr, box, cv::Scalar(255,255,255), 2);
        cv::circle(bgr, cent, 3, cv::Scalar(255,255,255), -1);
        cv::putText(bgr, t.id, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255,255,255), 1);
      }
    }

    // Publish debug image
    auto out_msg = cv_bridge::CvImage(msg->header, "bgr8", bgr).toImageMsg();
    debug_pub_.publish(out_msg);
  }

  void onDepth(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (!detections_.size() || !has_info_) return;
    cv::Mat depth = cv_bridge::toCvShare(msg)->image;

    for (auto &d : detections_) {
      float z = depth.at<float>(d.centroid.y, d.centroid.x);
      float x = (d.centroid.x - cx_) * z / fx_;
      float y = (d.centroid.y - cy_) * z / fy_;

      mycobot_interfaces::msg::DetectedObject out;
      out.id     = d.id;
      out.xmin   = d.box.x;
      out.ymin   = d.box.y;
      out.width  = d.box.width;
      out.height = d.box.height;
      out.position.x = x;
      out.position.y = y;
      out.position.z = z;
      det_pub_->publish(out);
      RCLCPP_INFO(this->get_logger(),
                  "Detected %s at [%.2f, %.2f, %.2f]", 
                  d.id.c_str(), x, y, z);
    }
    detections_.clear();
  }

  bool has_info_;
  float fx_, fy_, cx_, cy_;

  image_transport::Subscriber    image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<mycobot_interfaces::msg::DetectedObject>::SharedPtr det_pub_;
  image_transport::Publisher     debug_pub_;

  std::vector<Detection> detections_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ColorDepthDetector>());
  rclcpp::shutdown();
  return 0;
}
