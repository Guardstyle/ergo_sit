# Import Needed Library
import cv2
import math as m
import mediapipe as mp
import pygame

class PostureChecker:

    def __init__(self):
        # Initialize mediapipe pose class.
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        # Initialize Colors
        self.blue = (255, 127, 0)
        self.red = (50, 50, 255)
        self.green = (127, 255, 0)
        self.dark_blue = (127, 20, 0)
        self.light_green = (127, 233, 100)
        self.yellow = (0, 255, 255)
        self.pink = (255, 0, 255)

        # Font type.
        self.font = cv2.FONT_HERSHEY_SIMPLEX


    # Euclidean Distance
    def dist(self, x1, y1, x2, y2):
        res = m.sqrt((x1-x2)**2 + (y1-y2)**2)
        return res

    # Find Angle
    def angle(self, x1, y1, x2, y2):
        a = (y2 - y1) * (-y1)
        b = (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1)
        if b != 0:
            theta = m.acos(a / b)
            degree = int(180 / m.pi) * theta
        else:
            degree = 0
        return degree

    # Send warning if User is on bad posture for too long
    def warning(self, image, h, w):
        # Warning text 
        text = "Please Correct Your Posture"

        # get boundary of this text
        textsize = cv2.getTextSize(text, self.font, 1, 2)[0]

        # get coords based on boundary
        textX = (w - textsize[0]) // 2
        textY = (h + textsize[1]) // 2

        cv2.putText(image, text, (textX, textY), self.font, 1, self.red, 2)

    def run(self, file_name = 0):
        capt = cv2.VideoCapture(file_name)

        good_frames = 0
        bad_frames = 0

        # Meta
        fps = int(capt.get(cv2.CAP_PROP_FPS))
        width = int(capt.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capt.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Video writer
        vid_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

        # Declare flag so alarm didn't played repeatedly
        flag = True

        # Declare Pygame for alarm
        pygame.init()
        pygame.mixer.music.load("alarm.mp3")

        while capt.isOpened():
            success, image = capt.read()

            if not success:
                print("Null.Frames")
                break
            
            # Get fps
            fps = capt.get(cv2.CAP_PROP_FPS)

            # Get image height and width
            h, w = image.shape[:2]

            # =============================
            # Check Posture
            # =============================

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            keypoints = self.pose.process(image)
            lm = keypoints.pose_landmarks
            lmPose = self.mp_pose.PoseLandmark

            # Convert the image back to BGR.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # No human on the frame
            if lm == None:
                cv2.putText(image, "No Human Detected", (10, h - 20), self.font, 0.9, self.yellow, 2)

                # Write frames.
                vid_output.write(image)

                # Display.
                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                continue
            
            # Acquire landmark coordinates
            # Left Shoulder
            left_shoulder_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            left_shoulder_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * w)
            # Right Shoulder
            right_shoulder_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            right_shoulder_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * w)
            # Left Ear
            left_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            left_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * w)
            # Left hip
            left_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            left_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * w)

            # Calculate angles
            neck_inclination = self.angle(left_shoulder_x, left_shoulder_y, left_ear_x, left_ear_y)
            torso_inclination = self.angle(left_hip_x, left_hip_y, left_shoulder_x, left_shoulder_y)

            offset = self.dist(left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y)
            
            if offset < 100:
                cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), self.font, 0.5, self.green, 2)
            else:
                cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), self.font, 0.5, self.red, 2)            

            # Draw landmarks.
            cv2.circle(image, (left_shoulder_x, left_shoulder_y), 7, self.yellow, -1)
            cv2.circle(image, (left_ear_x, left_ear_y), 7, self.yellow, -1)
            cv2.circle(image, (left_shoulder_x, left_shoulder_y - 100), 7, self.yellow, -1)
            cv2.circle(image, (right_shoulder_x, right_shoulder_y), 7, self.pink, -1)
            cv2.circle(image, (left_hip_x, left_hip_y), 7, self.yellow, -1)
            cv2.circle(image, (left_hip_x, left_hip_y - 100), 7, self.yellow, -1)

            # Put text, Posture and angle inclination.
            angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

            color = color2 = None

            if neck_inclination < 25 and torso_inclination < 8:
                bad_frames = 0
                good_frames += 1

                color = self.light_green
                color2 = self.green
            else:
                good_frames = 0
                bad_frames += 1

                color = self.red
                color2 = self.red

            cv2.putText(image, angle_text_string, (10, 30), self.font, 0.9, color, 2)
            cv2.putText(image, str(int(neck_inclination)), (left_shoulder_x + 10, left_shoulder_y), self.font, 0.9, color, 2)
            cv2.putText(image, str(int(torso_inclination)), (left_hip_x + 10, left_hip_y), self.font, 0.9, color, 2)

            # Join landmarks.
            cv2.line(image, (left_shoulder_x, left_shoulder_y), (left_ear_x, left_ear_y), color2, 4)
            cv2.line(image, (left_shoulder_x, left_shoulder_y), (left_shoulder_x, left_shoulder_y - 100), color2, 4)
            cv2.line(image, (left_hip_x, left_hip_y), (left_shoulder_x, left_shoulder_y), color2, 4)
            cv2.line(image, (left_hip_x, left_hip_y), (left_hip_x, left_hip_y - 100), color2, 4)

            # Calculate the time of remaining in a particular posture.
            good_time = (1 / fps) * good_frames
            bad_time =  (1 / fps) * bad_frames

            # Pose time.
            if good_time > 0:
                if not flag:
                    pygame.mixer.music.stop()
                    flag = True
                time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                cv2.putText(image, time_string_good, (10, h - 20), self.font, 0.9, self.green, 2)
            else:
                time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                cv2.putText(image, time_string_bad, (10, h - 20), self.font, 0.9, self.red, 2)

            # If you stay in bad posture for more than 10 seconds, send an alert.
            if bad_time > 10:
                self.warning(image, h, w)
                
                if flag:
                    pygame.mixer.music.play()
                    flag = False

            # Write frames.
            vid_output.write(image)

            # Display.
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        capt.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    pos = PostureChecker()

    pos.run(0)
