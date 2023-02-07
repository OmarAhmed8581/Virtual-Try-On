from flask import * 
import mediapipe as mp
import cv2
import os
from werkzeug.utils import secure_filename
import shutil



use_brect = True
images_flat="1"

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg','png','jfif'])
cap = None
images = None
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose 
mp_hands = mp.solutions.hands


hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


mode = 0
count_store=0
UPLOAD_FOLDER = 'input'
UPLOAD_Video = 'input'


# print(mp_holistic.FACEMESH_CONTOURS)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_Video'] = UPLOAD_Video
# cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/get_login",methods=["get","post"])
def get_login():
    email = request.args.get('username')
    password=request.args.get('password')
    if email == "admin" and password =="admin":
        return redirect(url_for("home"))
    return redirect(url_for("login"))

@app.route("/")
def home():
    global images_flat,count_store
    images_flat="1"
    count_store=0

    # cache.clear()
    file_list = os.listdir("static/gallery")
    remove_store_folder()

    return render_template("index.html",file_name = file_list,value=len(file_list))


def change_background_color(images):
    
    
    t_shirt_gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    thesh_value = 0
    print("<------------- Image starting value")
    print(images[0][0][0])
    if images[0][0][0]==255:
        thesh_value = 242
    elif images[0][0][0]==0:
        thesh_value = 20
    else:
        thesh_value = images[0][0][0] - 13
    ret, mask = cv2.threshold(t_shirt_gray, thesh_value, 255, cv2.THRESH_BINARY)
    thesh = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))  
    cv2.floodFill(thesh, None, seedPoint=(0, 0), newVal=128, loDiff=1, upDiff=1)  
    images[thesh == 128] = (255, 255, 255)
    return images
    
# load a images images
def loadImages():
    folder = "input"
    images = []
    thres = [40, 75, 130, 130];
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


# Check the image format -------------------------------------------->
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/process_images",methods=["get","post"])
def process_images():
    global images
    # cache.clear()
    msg=""
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request','status':False})
        resp.status_code = 400
        return resp
    files = request.files.getlist('files[]')
    errors = {}
    filename=""
    if files[0] and allowed_file(files[0].filename):
        filename = secure_filename(files[0].filename)
        files[0].save(os.path.join(app.config['UPLOAD_Video'],"1.jpg"))   

    else:
        errors[files[0].filename] = 'File type is not allowed'
        msg = 'File type is only allowed (jpg,jpeg,png,jfif)'
    images = loadImages()
    if len(images)>0:
        images = change_background_color(images[0])
        msg= 'File save now open webcam'
        status= True
    else:
        msg="Image cannot read please select another images"
        status= False

    return  jsonify({
        'message' : msg,
        'status': status
        })


@app.route("/process_Video",methods=["get","post"])
def process_Video():
    global cap,images
    # open camera
    cap = cv2.VideoCapture(0)
    # cache.clear()
    return  jsonify({'message' : 'File save'})

@app.route('/video_feed')
def video_feed():

    return Response(webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/get_images",methods=["get","post"])
def get_images():
    value = len(os.listdir("./static/store"))
    if value==0:
        data = ""
    else:
        data=os.listdir("./static/store")[0]
    
    return jsonify({
        "len":value,"data":data
    })

def remove_store_folder():
    dir = 'static/store/'

    for f in os.listdir(dir):
        os.remove(os.path.join(dir,f))

@app.route("/save_images",methods=["get","post"])
def save_images():
    global images_flat,count_store
    images_flat="1"
    # cache.clear()
    value = len(os.listdir("./static/gallery"))
    src_path = './static/store/store_'+str(count_store)+'.jpg'
    count_store+=1
    dst_path = './static/gallery/'
    shutil.copy(src_path, dst_path + str(value)+'.jpg')
    # remove_store_folder()
   
    file_list = os.listdir("static/gallery")

    return jsonify({
        "msg":"Successfull save",
        "count":len(file_list)-1
    })


@app.route("/remove_images",methods=["get","post"])
def remove_images():
    global images_flat
    images_flat="1"
    # count_store +=1
    # cache.clear()
    # remove_store_folder()
    return jsonify({
        "msg":"Successfull remove"
    })



def webcam():
    global cap,images,images_flat,count_store
    
    size = 180
    mode = 0
    count=len(os.listdir("static/gallery"))

    images_flat="1"

# Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            _, frame = cap.read()
            frame_copy = frame
            
            image_height, image_width, _ = frame.shape
            frame = cv2.flip(frame, 1)
                        
           
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Hand gesture

            className = ''

            # post process the result
           
            fingerCount = 0

            tshirt_status=False

            # hand gesture

            if result.multi_hand_landmarks:

                for hand_landmarks in result.multi_hand_landmarks:
                    # Get hand index to check label (left or right)
                    handIndex = result.multi_hand_landmarks.index(hand_landmarks)
                    handLabel = result.multi_handedness[handIndex].classification[0].label

                    # Set variable to keep landmarks positions (x and y)
                    handLandmarks = []

                    # Fill list with x and y positions of each landmark
                    for landmarks in hand_landmarks.landmark:
                        handLandmarks.append([landmarks.x, landmarks.y])

                    # Test conditions for each finger: Count is increased if finger is 
                    #   considered raised.
                    # Thumb: TIP x position must be greater or lower than IP x position, 
                    #   deppeding on hand label.
                    if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                        fingerCount = fingerCount+1
                    elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                        fingerCount = fingerCount+1

                    # Other fingers: TIP y position must be lower than PIP y position, 
                    #   as image origin is in the upper left corner.
                    if handLandmarks[8][1] < handLandmarks[6][1]:       #Index finger
                        fingerCount = fingerCount+1
                    if handLandmarks[12][1] < handLandmarks[10][1]:     #Middle finger
                        fingerCount = fingerCount+1
                    if handLandmarks[16][1] < handLandmarks[14][1]:     #Ring finger
                        fingerCount = fingerCount+1
                    if handLandmarks[20][1] < handLandmarks[18][1]:     #Pinky
                        fingerCount = fingerCount+1

                    # Draw hand landmarks 
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            

            try:

                # body pose working
                right_shoulder=12
                right_shoulder_visibility = results.pose_landmarks.landmark[mp_pose.PoseLandmark(right_shoulder).value].visibility
                right_shoulder_x = int((results.pose_landmarks.landmark[mp_pose.PoseLandmark(right_shoulder).value].x * image_width)-45)
                right_shoulder_y = int((results.pose_landmarks.landmark[mp_pose.PoseLandmark(right_shoulder).value].y * image_height)-40)
        


                left_shoulder=11
                left_shoulder_visibility = results.pose_landmarks.landmark[mp_pose.PoseLandmark(left_shoulder).value].visibility
                left_shoulder_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark(left_shoulder).value].x * image_width)
                left_shoulder_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark(left_shoulder).value].y * image_height)


                right_hip=24
                right_hip_visibility = results.pose_landmarks.landmark[mp_pose.PoseLandmark(right_hip).value].visibility
                right_hip_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark(right_hip).value].x * image_width)
                right_hip_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark(right_hip).value].y * image_height)

                left_hip=23
                left_hip_visibility = results.pose_landmarks.landmark[mp_pose.PoseLandmark(left_hip).value].visibility
                left_hip_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark(left_hip).value].x * image_width)
                left_hip_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark(left_hip).value].y * image_height)

                print("<--------------------   Chest body ------------------------------->")
                print(f"right_shoulder_visibility {0}",right_shoulder_visibility)
                print(f"left_shoulder_visibility {0}",left_shoulder_visibility)
                print(f"right_hip_visibility {0}",right_hip_visibility)
                print(f"left_hip_visibility {0}",left_hip_visibility)
                # tshirt_status=False

                if (right_shoulder_visibility>0.1 and left_shoulder_visibility>0.1 and right_hip_visibility>0.1 and left_hip_visibility>0.1):
                    try:


                        # shirt size adjust

                        # right_eyes=5
                        # print(f'{mp_pose.PoseLandmark(right_eyes).name}:') 
                        # right_eyes_x = int((results.pose_landmarks.landmark[mp_pose.PoseLandmark(right_eyes).value].x * image_width))
                        # right_eyes_y = int((results.pose_landmarks.landmark[mp_pose.PoseLandmark(right_eyes).value].y * image_height))

                        t_shirt = images
                        # cal = (right_eyes_x - right_eyes_y) - (left_shoulder_x-left_shoulder_y)
                       
                        dis_width = abs((left_shoulder_x - right_shoulder_x))+50

                        # dis_height =int(math.sqrt((left_hip_y-left_shoulder_x)**2 + (left_hip_x-left_shoulder_x)**2))

                        dis_height =abs((left_shoulder_x - left_hip_y))+240


                        t_shirt =  cv2.resize(t_shirt, (dis_width,dis_height),
                            interpolation = cv2.INTER_LINEAR)
                        
                       
                        #   shirt body place and background remove

                        f_height = frame.shape[0]
                        f_width = frame.shape[1]
                        t_height = t_shirt.shape[0]
                        t_width = t_shirt.shape[1]

                        t_shirt_gray = cv2.cvtColor(t_shirt, cv2.COLOR_BGR2GRAY)
                        _, mask = cv2.threshold(t_shirt_gray, 251, 255, cv2.THRESH_BINARY_INV)
                        mask_inv = cv2.bitwise_not(mask)
                        roi = frame[right_shoulder_y:right_shoulder_y + t_height, right_shoulder_x:right_shoulder_x + t_width]
                        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                        img_fg = cv2.bitwise_and(t_shirt, t_shirt, mask=mask)
                        t_shirt = cv2.add(img_bg, img_fg)
                        tshirt_status=True
                        frame[right_shoulder_y:right_shoulder_y + t_height, right_shoulder_x:right_shoulder_x + t_width] = t_shirt
                        
                        # frame_copy[right_shoulder_y:right_shoulder_y + t_height, right_shoulder_x:right_shoulder_x + t_width] = t_shirt
                        
                        
                        
                    except:
                        print("error......................")
                        pass
            except:
                pass
            # cv2.putText(frame, str(fingerCount), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            print("images flat"+str(images_flat))       
            if tshirt_status:
                if str(fingerCount)=='1':
                    # if images_flat=='1':
                    fingerCount="capture"
                    images_flat="0"
                    cv2.putText(frame, str(fingerCount), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    cv2.imwrite("static/store/store_"+str(count_store)+".jpg",frame)
                    
                else:
                    cv2.putText(frame, str(fingerCount), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            
            # Draw face landmarks
            # mp_drawing.draw_landmarks(debug_image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            
            # Right hand
            # mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Left Hand
            # mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            # print("<---------------- pose_landmarks -------------------------->")
            # print(mp_holistic.POSE_CONNECTIONS)
            # Pose Detections
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            # cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
          
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                            
            

if __name__ == '__main__':
    app.run(debug=True) 