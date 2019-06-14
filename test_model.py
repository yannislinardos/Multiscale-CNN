from utils import *


if __name__ == "__main__":

    X, Y = load_data('Dataframes/Testing24.pickle')
    # load weights into new model
    model = load_model('Models/Model_24KHz_80%.yaml', 'Models/Model_24KHz_80%.h5')

    # score = test_model(model, X, Y)
    # print("%s: %.2f%%" % (model.metrics_names[1], score))
    #

