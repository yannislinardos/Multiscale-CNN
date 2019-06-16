from utils import *
from keras import backend as K

if __name__ == "__main__":

    # cfg = K.tf.ConfigProto()
    # cfg.gpu_options.allow_growth = True
    # K.set_session(K.tf.Session(config=cfg))

    X, Y = load_data('Dataframes/Testing24.pickle')
    # load weights into new model
    # model = load_model('Models/Multiscaled/test.yaml', 'Models/Multiscaled/test.h5')

    model = load_model('Models/Model_24KHz_87%_meanpooling.yaml', 'Models/Model_24KHz_87%_meanpooling.h5')

    score = test_model(model, X, Y)
    print("%s: %.2f%%" % (model.metrics_names[1], score))


