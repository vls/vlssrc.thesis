from PIL import Image

def main():
    imn = Image.new('P', (28, 28), 255)
    imn.putpixel((2,3), 144)
    imn.save('test.bmp', 'BMP')
    
if __name__ == '__main__':
    main()