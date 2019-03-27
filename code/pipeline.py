from feature_extraction import*





def main():
    image1 = read_img('dataset/Bicycle/images/0000.jpg')
    image2 = read_img('dataset/Bicycle/images/0020.jpg')
    extract_and_match(image1, image2)










if __name__ == "__main__":
    main()
