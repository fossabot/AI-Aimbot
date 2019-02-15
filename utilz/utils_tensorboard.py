import tensorflow as tf
import keras
import skimage

def make_image(tensor):
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag):
        super().__init__() 
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):

        # Load random image from dataset
        
        img = data.astronaut()
        # Do something to the image
        img = (255 * skimage.util.random_noise(img)).astype('uint8')

        #load image sample from training

        #run through model
        self.model.predict(img)

        image = make_image(img)
        image_predicted = make_image(img2)

        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image),tf.Summary.Value(tag=self.tag, image=image_predicted) ])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, epoch)
        writer.close()

        return

tbi_callback = TensorBoardImage('Image Example')