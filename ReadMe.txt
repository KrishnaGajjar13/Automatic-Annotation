There will be 6 views of an object each view is going to be taken as mode.
L stands for length, B stands for breadth, W stands for width, dimension used for annotation in each view will be

1. Front View (L, B)
2. Back View (L, B)
3. Top View (W, B)
4. Bottom View (W, B)
5. Left-Hand Side View (L, W)
6. Right-Hand Side View (L, W)

We will need to consider the distance ietween the camera and the object we will use 3 distances: 30, 65, and 100 cm for normal-sized objects, for large objects roughly we will start from 50, 90, and 130 cm, and for small objects 20, 50, 80 cm are the distance we will be using.

For Cylindrical Objects, Width and Breadth are going to be the same. So we can classify it as a separate condition. For circular objects, we will take radius as a parameter and use it to determine bounded box length probably the equation for that will be bounded box l is equal to 2*radius of the object.

Now positioning of objects should be aligned with the exact position of the camera, putting the object at the center of the image for now we are testing it on the table later we might mount it on top upside down facing the object.

Now for model training, we might need anything between 500-1000 images per class whereas later we will increase image count by applying complex image augmentation to increase the count to up to 2000-3000 images which is the recommended amount for image detection. Image reshaping is going to be 512*512 We are not going for standard 300 * 300 as there will be a variety of objects with small changes in different text, colors, sizes, etc.

We will take 10-second videos for each mode converting them into 100-150 per mode for a total of 6 modes making them 600-900 images. Later we might do some trial and error with these amounts. 

For the Logitech C525 webcam, with a 1920x1080 resolution and a 69° diagonal FOV and webcam angle is at 102°:

Pixels per centimeter (PPCM) for width: 36.23 pixels/cm.
Pixels per centimeter (PPCM) for height: 27.0 pixels/cm.	

The focal Length of the webcam is 3.64583 where the equation to estimate length is Object Length in pixels = (Object Length * Distance between camera and object)/3.64583. Here length can be either width height (breadth or radius) for one of the 6 views dimensions will be considered. 

The size of the cart will be less than 70-80 cm so there is no need to take images of objects far from that distance. For large objects like heights exceeding 30-35 cm then we will extend to 100 cm.

Now the classification of the sizes there will be 3 categories of objects based on size, height less than 10 cm will be captured at (20, 60), height less than 20 cm will be captured at (40, 75), and objects with height up to 30 cm can be added for detection images will be captured at (50,85). W.R.T. to Spherical object for each class of object radius should be half of height so for small radius should be < 5 cm, 10 cm, and 15 cm.

Bottom padding
20 cm - 25-pixel padding 
-> Center of object at center will be at (960,1055) + or - 1/2 * (Breadth/width) for cuboid. For cylindrical or spherical objects (960,1055) + or - (Radius of object).

40 cm - 115-pixel padding 
-> Center of object at center will be at (960,965) + or - 1/2 * (Breadth/width) for cuboid. For cylindrical or spherical objects, (960,965) + or - (Radius of object).

50 cm - 190-pixel padding 
-> Center of object at center will be at (960,890) + or - 1/2 * (Breadth/width) for cuboid. For a cylindrical or spherical object (960,890) + or - (Radius of the object).

60 cm - 
-> The center of the object at the center will be at (960,) + or - 1/2 * (Breadth/width) for the cuboid. For cylindrical or spherical object (960,) + or - (Radius of the object).

75 cm -
-> The center of the object at the center will be at (960,) + or - 1/2 * (Breadth/width) for the cuboid. For cylindrical or spherical object (960,) + or - (Radius of the object).

85 cm -
-> The center of the object at the center will be at (960,) + or - 1/2 * (Breadth/width) for the cuboid. For cylindrical or spherical object (960,) + or - (Radius of the object).

Equation for focal_length_pixel is diagonal = (width^2+height^2)^0.5 
focal_length_pixel = diagonal/(2* tan (fov/2)) where fov is in radians for 1000 by 1000 pixel value is 1414.21

