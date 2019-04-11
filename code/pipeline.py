from feature_extraction import*
from feature_matching import*
from geometric_verification import*
#from 3d_reconstruction import*

class Pair:
    img_inx_1 = -1
    img_inx_2 = -1
    matches = []


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        images.append(img)
        #if img is not None:
            #images.append(img)
    images = np.array(images)
    return images


def main():
    #Read images
    images = load_images_from_folder('dataset/Bicycle/images')
    print(images.shape)

    #.............................Feature Extraction and Matching......................

    #Example
    #image1 = read_img('dataset/Bicycle/images/0000.jpg')
    #image2 = read_img('dataset/Bicycle/images/0020.jpg')

    #matches_example = extract_and_match_draw(image1, image2)

    pair_matches = []
    count = 0
    for i in range(images.shape[0]):
        for j in range(images.shape[0]):
            if i != j:
                match = extract_and_match(images[i], images[j])
                if match:
                    pair = Pair()
                    pair.img_inx_1 = i
                    pair.img_inx_2 = j
                    pair.matches = match
                    pair_matches.append(pair)
            count += 1
            print(count)
    print(len(pair_matches))
    print(pair_matches[0].img_inx_1)
    print(pair_matches[0].img_inx_2)
    print(pair_matches[0].matches)






    #..........................Geometric Verification...................................






if __name__ == "__main__":
    main()
