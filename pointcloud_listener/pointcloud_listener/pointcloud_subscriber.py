import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

class GroundPointExtractor(Node):
    def __init__(self):
        super().__init__('ground_point_extractor')

        self.subscription = self.create_subscription(
            PointCloud,
            '/carmaker/pointcloud',
            self.pointcloud_callback,
            20
        )

        self.markers_publisher = self.create_publisher(MarkerArray, '/cluster_cylinders', 20)

        self.frame_id = 'map'
        self.z_threshold = 0.03  # +/- 3 cm from ground plane is considered "on ground"
        self.get_logger().info("Subscribed to /carmaker/pointcloud")

    def pointcloud_callback(self, msg):
        points = np.array([[p.x, p.y, p.z] for p in msg.points])

        if points.ndim != 2 or points.shape[1] != 3:
            self.get_logger().warn("Invalid point format.")
            return

        # Fit a ground plane using RANSAC (z = f(x, y))
        X = points[:, :2]  # x, y
        y = points[:, 2]   # z

        poly = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(X)
        ransac = RANSACRegressor(residual_threshold=self.z_threshold)
        ransac.fit(X_poly, y)

        # Predict ground height and compute residuals
        z_pred = ransac.predict(X_poly)
        residuals = np.abs(y - z_pred)

        # Keep only ground-level points (within z_threshold)
        ground_mask = residuals < self.z_threshold
        ground_points = points[ground_mask]

        self.get_logger().info(f"Total: {len(points)}, Ground: {len(ground_points)}")

        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        if len(ground_points) > 0:
            clustering = DBSCAN(eps=0.2, min_samples=5).fit(ground_points[:, :2])
            labels = clustering.labels_
            unique_labels = set(labels)
            num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            self.get_logger().info(f"DBSCAN found {num_clusters} clusters")

            marker_id = 0
            for cluster_id in unique_labels:
                if cluster_id == -1:
                    continue

                cluster_points = ground_points[labels == cluster_id]
                if len(cluster_points) < 3:
                    continue  # Too small, skip noise

                centroid = np.mean(cluster_points, axis=0)
                marker = Marker()
                marker.header.frame_id = self.frame_id
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "cone"
                marker.id = marker_id
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.31  # Cone height
                marker.pose.position.x = float(centroid[0])
                marker.pose.position.y = float(centroid[1])
                marker.pose.position.z = marker.scale.z / 2
                marker.pose.orientation.w = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                marker_array.markers.append(marker)
                marker_id += 1
        else:
            self.get_logger().info("No valid ground points found.")

        self.markers_publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = GroundPointExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
