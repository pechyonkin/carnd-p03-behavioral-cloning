# global parameters
# global parameters
angle_correction = 0.2
validation_split = 0.2
batch_size = 256
model_debug = False # makes a simple model for debug
udacity_data = False
loading_model = True
debug_cleaning = False

if loading_model:
    model_name = 'model.h5' # trained for 20 epochs

# imports
import os, platform, glob, csv, cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
%matplotlib inline

# routines
def get_lines(path_to_log):
    '''
    this routine returns list lines from a csv file 
    '''
    lines = []
    with open(path_to_log) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def get_image_path(log_path, path_to_imgs):
    '''
    returns an image path by appending filename from log_path
    to the images directory from path_to_imgs
    '''
    image_path = os.path.join(path_to_imgs, log_path.split('/')[-1])
    return image_path

def get_image(log_path, path_to_imgs):
    '''
    this routine returns an image by appending filename from log_path
    to the images directory from path_to_imgs
    '''
    image_path = get_image_path(log_path, path_to_imgs)
    # cv2 reads to BGR
    return cv2.imread(image_path)

def normalize_pixels(pixel_value):
    '''
    normalizes pixel values to have zero mean and SD of 1
    '''
    result = pixel_value / 255.0 - 0.5
    return result

def print_line(line):
    '''
    prints a sample line from csv
    for debugging purposes
    '''
    print("-"*15)
    print("Printing a sample line:")
    for n, e in zip(lines_headers, line):
        print("\t{}: {}".format(n, e))
    print("-"*15)
    
def make_labels_filenames():
    '''
    returns file paths to all images from log file
    as well as all corresponding steering angles
    steering angles are corrected by a correction factor
    defined globally
    '''
    labels = []
    filenames = []
    
    for i in range(nb_imgs):
        line = lines[i]
        
        center_image_path = get_image_path(line[0], path_to_imgs)
        left_image_path = get_image_path(line[1], path_to_imgs)
        right_image_path = get_image_path(line[2], path_to_imgs)
        
        center_angle = float(line[3])
        left_angle = center_angle + angle_correction
        right_angle = center_angle - angle_correction
        
        labels.extend([center_angle, left_angle, right_angle])
        filenames.extend([center_image_path, left_image_path, right_image_path])
        
    return filenames, labels
    
def clean_data(data, labels, depth=0, n_bins=20, debug=False):
    '''
    this routine removes data in overrepresented bins in a histogram of steering angles
    depth paratemeter tells what is the target number of examples in each bin
    depth of 0 means all bins should have at most as many elements as the second largest bin
    depth of 1 means all bins should have at most as many elements as the third largest bin etc.
    returns modified images and labels
    ---
    superceded by clean_filenames_labels which is a generator and works faster
    by not storing actual data in memory
    '''
    (counts, bins, _) = plt.hist(labels, bins=np.linspace(-1,1,n_bins), label='hst')
    
    target_size = int(sorted(counts, reverse=True)[1 + depth])
    if debug: print("Target bin count:", target_size)
    if debug: print("-"*15)
    
    all_indices_to_remove = []
    
    for pos, count in enumerate(counts):
        if debug: print("Position:", pos)
        if debug: print("Count:", count)
        if count > target_size:
            if debug: print("Not alright, need to remove some.")
            lower_bound = bins[pos]
            upper_bound = bins[pos + 1]
            if debug: print("Lower:", lower_bound)
            if debug: print("Upper:", upper_bound)
            all_indices = np.where((labels > lower_bound) & (labels < upper_bound))[0]
            n_to_remove = len(all_indices) - target_size
            if debug: print("Need to remove:", n_to_remove)
            indices_to_remove = np.random.choice(all_indices, size = n_to_remove, replace=False)
            if debug: print("All indices before:", len(all_indices_to_remove))
            all_indices_to_remove.extend(indices_to_remove)
            if debug: print("All indices after:", len(all_indices_to_remove))
        else:
            if debug: print("Alright, alright, alright!")
            continue
        if debug: print("-"*15)

    labels_new = np.delete(labels, all_indices_to_remove, axis=0)
    data_new = np.delete(data, all_indices_to_remove, axis=0)
    return data_new, labels_new

def clean_filenames_labels(filenames, labels, depth=0, n_bins=20, debug=False):
    '''
    this routine removes filenames and labels in overrepresented bins in a histogram of steering angles
    depth paratemeter tells what is the target number of examples in each bin
    depth of 0 means all bins should have at most as many elements as the second largest bin
    depth of 1 means all bins should have at most as many elements as the third largest bin etc.
    returns modified filenames and labels
    
    '''
    (counts, bins, _) = plt.hist(labels, bins=np.linspace(-1,1,n_bins), label='hst')
    
    target_size = int(sorted(counts, reverse=True)[1 + depth])
    if debug: print("Target bin count:", target_size)
    if debug: print("-"*15)
    
    all_indices_to_remove = []
    
    for pos, count in enumerate(counts):
        if debug: print("Position:", pos)
        if debug: print("Count:", count)
        if count > target_size:
            if debug: print("Not alright, need to remove some.")
            lower_bound = bins[pos]
            upper_bound = bins[pos + 1]
            if debug: print("Lower:", lower_bound)
            if debug: print("Upper:", upper_bound)
            all_indices = np.where((labels > lower_bound) & (labels < upper_bound))[0]
            n_to_remove = len(all_indices) - target_size
            if debug: print("Need to remove:", n_to_remove)
            indices_to_remove = np.random.choice(all_indices, size = n_to_remove, replace=False)
            if debug: print("All indices before:", len(all_indices_to_remove))
            all_indices_to_remove.extend(indices_to_remove)
            if debug: print("All indices after:", len(all_indices_to_remove))
        else:
            if debug: print("[McConaughey voice] Alright, alright, alright!")
            continue
        if debug: print("-"*15)

    labels_new = np.delete(labels, all_indices_to_remove, axis=0)
    filenames_new = np.delete(filenames, all_indices_to_remove, axis=0)
    return filenames_new, labels_new 

def make_clean_data(filenames, labels):
    '''
    returns an array of filtered images stored in memory
    to be fed into custom data generator
    '''
    X = []
    for f in filenames:
        image = cv2.imread(f)
        X.append(image)
    X = np.array(X)
    print("Shape of X after cleaning is", X.shape)
    print("Shape of y after cleaning is", labels.shape)
    print()
    
    return X, labels

def make_model(act='elu', d=0.5, debug=False):
    '''
    nVidia end-to-end driving model
    1) custom activation and dropout
    2) custom preprocessing layers
    '''
    model = Sequential()
    model.add(Cropping2D(cropping=((65, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(normalize_pixels, output_shape=(70, 320, 3)))
    if debug:
        model.add(Flatten())
        model.add(Dense(1))
        return model
    model.add(Conv2D(24, (5,5), strides=(2,2), activation=act))
    model.add(Dropout(d))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation=act))
    model.add(Dropout(d))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation=act))
    model.add(Dropout(d))
    model.add(Conv2D(64, (3,3), activation=act))
    model.add(Dropout(d))
    model.add(Conv2D(64, (3,3), activation=act))
    model.add(Dropout(d))
    model.add(Flatten())
    model.add(Dense(100, activation=act))
    model.add(Dropout(d))
    model.add(Dense(50, activation=act))
    model.add(Dropout(d))
    model.add(Dense(10, activation=act))
    model.add(Dense(1))
    return model

def preprocess_image(image):
    # Gaussian blur
    image = cv2.GaussianBlur(image, (3,3), 0)
    # Convert to YUV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image

def brighten_image(image):
    '''
    performs random brightening / darkening of the image
    in order to generalize driving in different lighting
    conditions
    '''
    value = np.random.randint(-28, 28)
    # the mask prevents values from being outside (0,255)
    if value > 0:
        mask = (image[:,:,0] + value) > 255 
    if value <= 0:
        mask = (image[:,:,0] + value) < 0
    image[:,:,0] += np.where(mask, 0, value)
    return image

def shadow_image(image):
    '''
    random shadow to make the model drive on
    roads with random shadows
    shadow is random region:
        - full height
        - left or right portion of image
    shadow area will be 20%-40% darker
    '''
    height, width = image.shape[0:2]
    # random horizontal line
    mid = np.random.randint(0, width)
    # image is in YUV
    # factor darkens 1st channel (brightness)
    factor = np.random.uniform(0.6,0.8)
    # random shadow on the left or on the right of image
    if np.random.rand() > .5:
        image[:, 0:mid, 0] *= factor
    else:
        image[:, mid:width, 0] *= factor
    return image

def shift_horizon(image):
    '''
    randomly shift horizon to simulate
    driving in areas with hills
    this transform will move horizon
    vertically  up or down 
    for up to 1/8 of height
    '''
    height, width = image.shape[0:2]
    # horizon value (calculated empirically)
    horizon = 0.4 * height
    v_shift = np.random.randint(- height / 8, height / 8)
    pts1 = np.float32([[0, horizon], [width, horizon], [0, height], [width, height]])
    pts2 = np.float32([[0, horizon + v_shift], [width, horizon + v_shift], [0, height], [width, height]])
    transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    new_img = cv2.warpPerspective(image, transform_matrix, (width, height) , borderMode=cv2.BORDER_REPLICATE)
    return image

def augment_image(image, label, proba=0.5):
    ''' 
    method for adding random distortion to dataset images, including random brightness adjust, and a random
    vertical shift of the horizon position
    '''
    new_img = image.astype(float)
    # 1) randomly flip image horizontally and reverse label
    if np.random.rand() > proba:
        new_img = cv2.flip(new_img, 1)
        label = -label
    # 2) random brightness
    new_img = brighten_image(new_img)
    # 3) random shadow
    new_img = shadow_image(new_img)
    # 4) random horizon shift
    new_img = shift_horizon(new_img)
    return new_img.astype(np.uint8), label

def generate_training_data_from_disk(filenames, labels, batch_size=256, validation_flag=False):
    '''
    this is a generator that yields data from disk
    and performs preprocessing and augmentation on the fly,
    without storing all augmented data in memory
    if validation_flag is True, no augmentation is performed
    '''
    filenames, labels = shuffle(filenames, labels)
    X, y = ([], [])
    while True:
        for i in range(len(labels)):
            image = cv2.imread(filenames[i])
            label = labels[i]
            image = preprocess_image(image)
            if not validation_flag:
                image, label = augment_image(image, label)
            X.append(image)
            y.append(label)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([], [])
                filenames, labels = shuffle(filenames, labels)

def generate_training_data_from_memory(data_orig, labels_orig, batch_size=256, validation_flag=False, debug=False):
    '''
    this is a generator that yields data from numpy array in memory
    and performs preprocessing and augmentation on the fly,
    without storing all augmented data in memory
    if validation_flag is True, no augmentation is performed
    '''
    data = np.copy(data_orig)
    labels = np.copy(labels_orig)
    data, labels = shuffle(data, labels)
    if debug:
        original_image = data[0]
        preprocessed_image = preprocess_image(original_image)
        augmented_image = augment_image(preprocessed_image, labels[0])[0]
        yield (cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), 
               cv2.cvtColor(preprocessed_image, cv2.COLOR_YUV2RGB), 
               cv2.cvtColor(augmented_image, cv2.COLOR_YUV2RGB), 
               labels[0])
    X, y = ([], [])
    while True:
        for i in range(len(labels)):
            image = data[i]
            label = labels[i]
            image = preprocess_image(image)
            if not validation_flag:
                image, label = augment_image(image, label)
            X.append(image)
            y.append(label)
            if len(X) == batch_size:
                yield (np.array(X), np.array(y))
                X, y = ([], [])
                data, labels = shuffle(data, labels)
                
def report_images(save=False):
    '''
    this function plots and saves sample images for the report
    '''
    gen = generate_training_data_from_memory(X_clean, labels, batch_size=batch_size, debug=True)
    img1, img2, img3, lbl = next(gen)

    fig = plt.figure()
    plt.imshow(img1)
    if save: fig.savefig('images/01_original.png', bbox_inches='tight')

    fig = plt.figure()
    plt.imshow(img2)
    if save: fig.savefig('images/02_preprocessed.png', bbox_inches='tight')

    fig = plt.figure()
    plt.imshow(img3)
    if save: fig.savefig('images/03_augmented.png', bbox_inches='tight')
    
    img4 = img3[65:135,:,:]
    fig = plt.figure()
    plt.imshow(img4)
    if save: fig.savefig('images/04_cropped.png', bbox_inches='tight')
    
    img5 = normalize_pixels(img4)
    fig = plt.figure()
    plt.imshow(img5)
    if save: fig.savefig('images/05_normalized.png', bbox_inches='tight')
            

lines_headers = [
    'center',
    'left',
    'right',
    'angle',
    'throttle',
    'break',
    'speed'
]

# setting platform-specific parameters
if platform.system() == 'Linux':
    if udacity_data:
        path_to_imgs = "/home/fazzl/udacity/udacity-data/IMG"
        path_to_log = "/home/fazzl/udacity/udacity-data/driving_log.csv"
    else:
        path_to_imgs = "/home/fazzl/udacity/data/IMG"
        path_to_log = "/home/fazzl/udacity/data/driving_log.csv"
    path_to_model = "/home/fazzl/udacity/CarND-Behavioral-Cloning-P3"
    
    lines = get_lines(path_to_log)
    print_line(lines[0])
    
    # Full images on Linux GPU machine (for full training)
    nb_imgs = len(lines)
    print('Number of images is {}.'.format(nb_imgs))
elif platform.system() == 'Darwin':
    path_to_imgs = "/Users/Fazzl/Dropbox/data/IMG"
    path_to_log = "/Users/Fazzl/Dropbox/data/driving_log.csv"
    path_to_model = "/Users/Fazzl/Documents/MOOCs & Self Study Backed/CarND/car/term-1/CarND-Behavioral-Cloning-P3"
    
    lines = get_lines(path_to_log)
    print_line(lines[0])
    
    # Only 100 images on Darwin CPU machine (for testing code)
    nb_imgs = 100
    print('Number of images is {}.'.format(nb_imgs))


# ------------------------------------------------------- #
#               BELOW IS TRAIN CODE ITSELF                #
# ------------------------------------------------------- #

# make filenames and labels for using in generating batches
filenames, labels = make_labels_filenames()

# remove overrepresented labels and filenames
filenames, labels = clean_filenames_labels(filenames, labels, depth=2, debug=debug_cleaning)

# local data (stores in memory to be used in generator)
X_clean, labels_clean = make_clean_data(filenames, labels)

print("Used in data generators:")
print("X_clean shape:", X_clean.shape)
print("labels_clean shape:", labels_clean.shape)
print("-"*15)

# training, validation and testing generators
train_gen = generate_training_data_from_memory(X_clean, labels, batch_size=batch_size, validation_flag=False)
valid_gen = generate_training_data_from_memory(X_clean, labels, batch_size=batch_size, validation_flag=True)
test_gen = generate_training_data_from_memory(X_clean, labels, batch_size=batch_size, validation_flag=True)

# number of samples for validation and epoch should be multiple of batch size
number_valid_steps = int(validation_split * len(labels_clean) * 2) // batch_size
steps_per_epoch = int((1 - validation_split) * len(labels_clean) * 2) // batch_size
print("# of labels:", len(labels))
print("Batch size:", batch_size)
print("# valid samples:", number_valid_steps)
print("# per epoch:", steps_per_epoch)

# --------------------------------------------------------

# --------------------------------------------------------

# check histogram of the clean data
(n, bins, patches) = plt.hist(labels, bins=np.linspace(-1,1,20), label='hst')
plt.show()

# --------------------------------------------------------

if not loading_model:
    print("Making model:")
    model = make_model(debug=model_debug)
    model.compile(loss = 'mse', optimizer = 'adam')
    model.summary()
else:
    print("Loading model: ", model_name)
    model = load_model(path_to_model + '/' + model_name)
    model.summary()


# ------------------------------------------------------- #
#       TRAINING  AND SAVING THE MODEL                    #
# ------------------------------------------------------- #

model.fit_generator(train_gen, 
                    epochs=20,
                    validation_data=valid_gen, 
                    validation_steps=number_valid_steps, 
                    steps_per_epoch=steps_per_epoch, initial_epoch=0)

model.save(path_to_model + '/model.h5')