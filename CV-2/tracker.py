import cv2
import numpy as np
import sys
import os

def track_object(video_path, output_path=None):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    # Initialize SIFT detector
    # SIFT is invariant to scale and rotation, good for this task
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT on the first frame (the object)
    kp1, des1 = sift.detectAndCompute(first_frame, None)
    
    h_obj, w_obj = first_frame.shape[:2]
    # Define corners of the object (the whole first frame)
    obj_corners = np.float32([[0, 0], [0, h_obj-1], [w_obj-1, h_obj-1], [w_obj-1, 0]]).reshape(-1, 1, 2)

    # FLANN parameters for matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Prepare video writer if output path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write the first frame to output
    if out:
        # Draw the initial bounding box on the first frame?
        # Since the object is the whole frame, the box is the frame border.
        # Let's just write it as is or with a border.
        # For consistency, let's draw the border.
        first_frame_disp = first_frame.copy()
        cv2.polylines(first_frame_disp, [np.int32(obj_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(first_frame_disp, "Reference Object", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        out.write(first_frame_disp)

    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect and compute descriptors in the current frame
        kp2, des2 = sift.detectAndCompute(frame, None)

        if des2 is not None and len(des2) > 0:
            # Match descriptors
            matches = flann.knnMatch(des1, des2, k=2)

            # Store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            # If enough matches are found, we can find the object
            MIN_MATCH_COUNT = 10
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                # Find homography
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    # Project corners to new frame
                    dst_corners = cv2.perspectiveTransform(obj_corners, M)
                    
                    # Draw bounding box
                    frame = cv2.polylines(frame, [np.int32(dst_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    # Add label
                    x, y = np.int32(dst_corners[0][0])
                    cv2.putText(frame, "Tracked Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
                pass
        
        if out:
            out.write(frame)
        
        # Uncomment to show window if running locally with display
        # cv2.imshow('Tracking', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames. Output saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tracker.py <video_path> [output_path]")
    else:
        video = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else "output.avi"
        
        # Ensure directory exists
        out_dir = os.path.dirname(output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        track_object(video, output)

