import cv2, time,os, subprocess, csv
from PIL import Image as img
import xml.etree.ElementTree as ET
import numpy as np  # For NaN

mode = {1 : 'Front View',2 : 'Top View', 3 : 'Back View', 4 : 'Bottom View', 5 : 'Left-Hand Side View', 6 : 'Right-Hand Side View'}
pre_defined_shapes = {'A' : 'Cuboid', 'C':'Cylindrical', 'S': 'Spherical'}
cap = cv2.VideoCapture(2)
base = {
    20 : 35,
    40 : 60,
    50 : 70,
    60 : 85,
    75 : 100,
    85 : 100
}

def drawbox(i,shape,distance,*dimension):
    temp_base = int(base.get(distance)) # type: ignore
    focal_length_pixels = 1600
    y2 = 1080 - temp_base
    if shape == "Cuboid":
        pixel_height = (dimension[0] * focal_length_pixels)/distance
        pixel_breadth = (dimension[1] * focal_length_pixels)/distance
        pixel_width = (dimension[2] * focal_length_pixels)/distance
        match i:
            case 1 | 3:
                pixel_breadth = (pixel_breadth / 0.82)
                x1 = 960 - (pixel_breadth/2)
                y1 = 1080 - temp_base - pixel_height # x1,y1 are for Top left Corner
                x2 = 960 + (pixel_breadth/2)
                
            case 2 | 4:
                pixel_breadth = (pixel_breadth / 0.82)
                x1 = 960 - (pixel_breadth/2)
                y1 = 1080 - temp_base - pixel_width # x1,y1 are for Top left Corner
                x2 = 960 + (pixel_breadth/2)

            case 5 | 6:
                pixel_width = (pixel_width / 0.82)
                x1 = 960 - (pixel_width/2)
                y1 = 1080 - temp_base - pixel_height # x1,y1 are for Top left Corner
                x2 = 960 + (pixel_width/2)

    elif shape == "Cylindrical":
        pixel_height = (dimension[0] * focal_length_pixels)/distance
        pixel_radius = (dimension[1] * focal_length_pixels)/distance
        diameter = pixel_radius * 2
        pixel_radius = pixel_radius/0.8
        x1 = 960 - pixel_radius
        x2 = 960 + pixel_radius
        match i:
            case 1 | 3 | 5 | 6:
                y1 = 1080 - temp_base - pixel_height # x1,y1 are for Top left Corner
                
            case 2 | 4:
                y1 = 1080 - temp_base - diameter # x1,y1 are for Top left Corner
    else:
        pixel_radius = (dimension[0] * focal_length_pixels)/distance
        diameter = pixel_radius * 2
        pixel_radius = pixel_radius/0.8
        x1 = 960 - pixel_radius
        y1 = 1080 - temp_base - diameter
        x2 = 960 + pixel_radius
        
    
    # Draw a rectangle on the frame
    return int(x1),int(y1),int(x2),int(y2) # type: ignore
    
#print("Now lets capture images of object")
def get_videos(labelname, distance,shape,*dimension):
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')

    # Create the directory outside the loop
    os.makedirs(f'label_videos/{labelname}', exist_ok=True)
    for i in range(1, 7):
        # Define video writer object once inside the loop 
        out = cv2.VideoWriter(f'label_videos/{labelname}/{labelname}_{mode.get(i)}_{distance}.mp4', fourcc, 30.0, (1920,1080))

        # Record the start time
        start_time = None  # Set start time to None initially
        is_recording = False

        while True:
            # Read a frame from the camera
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture image.")
                break

            # Resize the frame to 1920x1080 (optional)
            frame = cv2.resize(frame, (1920,1080))

            # Display message before recording starts
            if not is_recording:
                color = (0, 255, 0)  # Green color (in BGR format)
                thickness = 2        # Thickness of the rectangle's border
                x1,y1,x2,y2 = drawbox(i,shape,distance,*dimension)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                text = f"Getting {mode[i]}, put object in center and press 's' for recording\nMake Sure Object is in box at given distance"
                position = (10, 30)  # Position in the top-left corner
                font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
                font_scale = 1  # Font scale
                color = (255, 255, 255)  # White color (BGR)
                thickness = 2  # Thickness of the text
                cv2.putText(frame, text, position, font, font_scale, color, thickness)

            # Display the resulting frame
            cv2.imshow('Camera', frame)

            # Check if 's' is pressed to start recording
            key = cv2.waitKey(3) & 0xFF
            if key == ord('s') and not is_recording:
                # Wait for 1 seconds (or any desired time) before starting recording
                wait_time = 3  # in seconds
                print(f"Starting recording in {wait_time} seconds...")
                time.sleep(wait_time)  # Sleep for the wait time
                is_recording = True
                start_time = time.time()  # Set the start time when 's' is pressed
                print("Recording started...")

            # If recording, save frames and check elapsed time
            if is_recording:
                out.write(frame)
                elapsed_time = time.time() - start_time # type: ignore
                if elapsed_time >= 10:  # Stop recording after 10 seconds
                    print("10 seconds elapsed. Stopping video capture.")
                    break

            # Check if 'e' is pressed to stop recording earlier
            if key == ord('e'):
                print("Recording stopped early.")
                break

            # Exit on pressing 'q'
            if key == ord('q'):
                exit()


def split_videos(distance, output_image_dir, input_video, labelname, pos):
        # Create the output directory if it doesn't exist
    os.makedirs(f'{output_image_dir}/{labelname}/', exist_ok=True)
    # FFmpeg command to extract frames at 15 fps
    ffmpeg_command = [
        "ffmpeg", 
        "-i", input_video,          # Input video file
        "-vf", "fps=15",          # Set the frame rate to 30 fps
        f"{output_image_dir}/{labelname}/{labelname}_{distance}_{pos}_%04d.jpg"  # Output image file format
    ]
    # Execute the FFmpeg command
    try:
        # Execute the FFmpeg command
        subprocess.run(ffmpeg_command, check=True)
        print(f"Frames extracted to {output_image_dir}/{labelname}_%04d.jpg")
    except subprocess.CalledProcessError as e:
        print(f"Error during frame extraction: {e}")

# Class to handle different shapes
class Shape:
    def __init__(self, label_name, shape,size, *dimensions):
        self.label_name = label_name
        self.shape = shape
        self.size = size
        # Store dimensions as a tuple
        self.dimensions = dimensions
    def get_shape_details(self):
    # Helper function to replace None with NaN
        def replace_none_with_nan(value):
            return value if value is not None else np.nan

        if self.shape == 'Cuboid':
            height, breadth, width = self.dimensions
            return {
                'label_name': self.label_name,
                'shape': self.shape,
                'size' : self.size,
                'height': replace_none_with_nan(height),
                'breadth': replace_none_with_nan(breadth),
                'width': replace_none_with_nan(width),
                'radius': np.nan
            }
        
        elif self.shape == 'Cylindrical':  
            height, radius = self.dimensions 
            return {
                'label_name': self.label_name,
                'shape': self.shape,
                'size' : self.size,
                'height': replace_none_with_nan(height),
                'breadth': np.nan,
                'width': np.nan,
                'radius': replace_none_with_nan(radius)
            }
        
        elif self.shape == 'Spherical':
            (radius,) = self.dimensions  # Ensuring it's correctly unpacked
            return {
                'label_name': self.label_name,
                'shape': self.shape,
                'size' : self.size,
                'height': np.nan,
                'breadth': np.nan,
                'width': np.nan,
                'radius': replace_none_with_nan(radius)
            }
        else:
            return {}

# Function to write shapes data to a CSV file
def write_shapes_to_csv(filename, shapes_data):
    fieldnames = ['label_name', 'shape','size', 'height', 'breadth', 'width', 'radius']

    # Check if file exists and is empty
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0  

    with open(filename, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header **only if the file is newly created or empty**
        if not file_exists:
            writer.writeheader()

        # Write each shape as a row
        for shape in shapes_data:
            writer.writerow(shape.get_shape_details())

shapes = []  # List to store multiple shape objects

def get_object(label_name, distance, shape, *dimension):
    #label_videos/krishna/20_krishna_1.mp4
    for i in range(2):
        print(f"Put your objects at {distance[i]} cm away from camera")
        get_videos(label_name,distance[i],shape , *dimension)
        for k in range(1,7):
            video_dir = f'label_videos/{label_name}/{label_name}_{mode.get(k)}_{distance[i]}.mp4'
            target_dir = 'label_images'
            split_videos(distance[i],target_dir, video_dir, label_name,pos = k)

# * * ---- From here code starts ----
while True:
    print("There are 3 shapes of the object A -> Cuboid, C -> Cylindrical, S -> Spherical")
    c = input("Enter shape of the object (or 'q' to quit): ").capitalize()

    if c == 'Q':  # Exit condition
        break

    if c not in ('A', 'C', 'S'):
        print("Invalid Input! Please enter A, C, or S.")
        continue
    
    print(f"Shape of the object is {pre_defined_shapes.get(c)}.")

    label_name = input("Enter Item Name: ")  # Prompt label name for each object
    print("All measurements should be in centimetre")
    if c == 'A':  # Cuboid
        while True:
            try:
                height = float(input("Enter the height of the object: "))
                breadth = float(input("Enter the breadth of the object: "))
                width = float(input("Enter the width of the object: "))
                if height > 0 and breadth > 0 and width > 0:
                    """if height < breadth or height < width or breadth < width:
                        print("Invalid dimensions! Ensure height > breadth > width.\n")
                        continue
                    else:"""
                    height, breadth, width = round(height, 2), round(breadth, 2), round(width, 2)
                    break
                else:
                    print("Please enter positive values for all dimensions.\n")
            except ValueError:
                print("Invalid input! Please enter numeric values.\n")
        
        if height <= 10.0:
            distance = (20,60) 
            Size = 'Small'
            get_object(label_name, distance,pre_defined_shapes.get(c),height, breadth, width)
            shapes.append(Shape(label_name, pre_defined_shapes.get(c),Size, height, breadth, width))
            
        elif height <= 20.0:
            Size = 'Medium'
            distance = (40,75)
            get_object(label_name, distance,pre_defined_shapes.get(c),height, breadth, width)
            shapes.append(Shape(label_name, pre_defined_shapes.get(c),Size, height, breadth, width))

        elif height <=35.0:
            Size = 'Large'
            distance = (50,90)
            get_object(label_name, distance,pre_defined_shapes.get(c),height, breadth, width)
            shapes.append(Shape(label_name, pre_defined_shapes.get(c),Size, height, breadth, width))
        else:
            print("Shape Out of Bounds")
            continue
        
    elif c == 'S':  # Sphere
        while True:
            try:
                radius = float(input("Enter the radius of the object: "))
                if radius > 0:
                    radius = round(radius, 2)
                    break
                else:
                    print("Please enter a positive value for the radius.\n")
            except ValueError:
                print("Invalid input! Please enter a numeric value.\n")

        if radius <= 5.0:
            distance = (20,60) 
            Size = 'Small'
            get_object(label_name, distance, pre_defined_shapes.get(c), radius)
            shapes.append(Shape(label_name, pre_defined_shapes.get(c), radius,Size))
            
        elif radius <= 10.0:
            Size = 'Medium'
            distance = (40,75)
            get_object(label_name, distance,pre_defined_shapes.get(c), radius)
            shapes.append(Shape(label_name, pre_defined_shapes.get(c), radius,Size))

        elif radius <=17.5:
            Size = 'Large'
            distance = (50,90)
            get_object(label_name, distance,pre_defined_shapes.get(c), radius)
            shapes.append(Shape(label_name, pre_defined_shapes.get(c), radius,Size))
        else:
            print("Shape Out of Bounds")
            continue

    else:  # Cylinder
        while True:
            try:
                height = float(input("Enter the height of the object: "))
                radius = float(input("Enter the radius of the object: "))

                if height > 0 and radius > 0:
                    """if height < radius:
                        print("Invalid dimensions! Height should be greater than radius.\n")
                        continue
                    else:"""
                    height, radius = round(height, 2), round(radius, 2)
                    break
                else:
                    print("Please enter positive values for both height and radius.\n")
            except ValueError:
                print("Invalid input! Please enter numeric values.\n")

        if height <= 10.0:
            distance = (20,60) 
            Size = 'Small'
            get_object(label_name, distance,pre_defined_shapes.get(c), height,radius)
            shapes.append(Shape(label_name, pre_defined_shapes.get(c), height,radius,Size))
            
        elif height <= 20.0:
            Size = 'Medium'
            distance = (40,75)
            get_object(label_name, distance,pre_defined_shapes.get(c), height,radius)
            shapes.append(Shape(label_name, pre_defined_shapes.get(c), height,radius,Size))

        elif height <=35.0:
            Size = 'Large'
            distance = (50,90)
            get_object(label_name, distance,pre_defined_shapes.get(c), height,radius)
            shapes.append(Shape(label_name, pre_defined_shapes.get(c), height, radius,Size))
        else:
            print("Shape Out of Bounds")
            continue

    # Ask if user wants to add another object
    cont = input("Do you want to add another object? (y/n): ").strip().lower()
    if cont not in ('y', 'yes'):
        break  # Exit loop if user chooses 'n' or 'no'


# Write all collected data to CSV at once
if shapes:
    write_shapes_to_csv('object_metadata.csv', shapes)
    print("Data successfully saved.")
else:
    print("No data was saved.")


# Release the capture and video writer when done
cap.release()
cv2.destroyAllWindows()
