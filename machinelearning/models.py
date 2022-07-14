import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        while True:
            All_Passed = 1
            print("start:")
            for x, y in dataset.iterate_once(batch_size):
                multiplier = self.get_prediction(x)
                #print(multiplier == nn.as_scalar(y))
                if multiplier != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    All_Passed = 0
                    print("wrong: mul="+str(multiplier)+"; y="+str(nn.as_scalar(y)))
            if All_Passed == 1:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.n0 = 1
        self.n1 = 50
        self.n2 = 30
        self.n3 = 20
        self.n4 = 1
        self.W1 = nn.Parameter(self.n0, self.n1)
        self.b1 = nn.Parameter(1, self.n1)
        self.W2 = nn.Parameter(self.n1, self.n2)
        self.b2 = nn.Parameter(1, self.n2)
        self.W3 = nn.Parameter(self.n2, self.n3)
        self.b3 = nn.Parameter(1, self.n3)
        self.W4 = nn.Parameter(self.n3, self.n4)
        self.b4 = nn.Parameter(1, self.n4)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        self.Z1 = nn.AddBias(nn.Linear(x, self.W1), self.b1)
        self.A1 = nn.ReLU(self.Z1)
        self.Z2 = nn.AddBias(nn.Linear(self.A1, self.W2), self.b2)
        self.A2 = nn.ReLU(self.Z2)
        self.Z3 = nn.AddBias(nn.Linear(self.A2, self.W3), self.b3)
        self.A3 = nn.ReLU(self.Z3)
        self.Z4 = nn.AddBias(nn.Linear(self.A3, self.W4), self.b4)
        self.predicted_y = self.Z4
        return self.predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        loss = nn.SquareLoss(predicted_y, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        multiplier = -0.05
        while True:
            
            loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            #print(str(nn.as_scalar(loss)))
            if nn.as_scalar(loss) < 0.02:
                break
            grads_W1, grads_b1, grads_W2, grads_b2, grads_W3, grads_b3, grads_W4, grads_b4 = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4])
            
            self.W1.update(grads_W1, multiplier)
            self.b1.update(grads_b1, multiplier)
            self.W2.update(grads_W2, multiplier)
            self.b2.update(grads_b2, multiplier)
            self.W3.update(grads_W3, multiplier)
            self.b3.update(grads_b3, multiplier)
            self.W4.update(grads_W4, multiplier)
            self.b4.update(grads_b4, multiplier)
            
                

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.n0 = 784
        self.n1 = 200
        self.n2 = 150
        self.n3 = 80
        self.n4 = 20
        self.n5 = 10
        self.W1 = nn.Parameter(self.n0, self.n1)
        self.b1 = nn.Parameter(1, self.n1)
        self.W2 = nn.Parameter(self.n1, self.n2)
        self.b2 = nn.Parameter(1, self.n2)
        self.W3 = nn.Parameter(self.n2, self.n3)
        self.b3 = nn.Parameter(1, self.n3)
        self.W4 = nn.Parameter(self.n3, self.n4)
        self.b4 = nn.Parameter(1, self.n4)
        self.W5 = nn.Parameter(self.n4, self.n5)
        self.b5 = nn.Parameter(1, self.n5)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        self.Z1 = nn.AddBias(nn.Linear(x, self.W1), self.b1)
        self.A1 = nn.ReLU(self.Z1)
        self.Z2 = nn.AddBias(nn.Linear(self.A1, self.W2), self.b2)
        self.A2 = nn.ReLU(self.Z2)
        self.Z3 = nn.AddBias(nn.Linear(self.A2, self.W3), self.b3)
        self.A3 = nn.ReLU(self.Z3)
        self.Z4 = nn.AddBias(nn.Linear(self.A3, self.W4), self.b4)
        self.A4 = nn.ReLU(self.Z4)
        self.Z5 = nn.AddBias(nn.Linear(self.A4, self.W5), self.b5)
        self.predicted_y = self.Z5
        return self.predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        loss = nn.SoftmaxLoss(predicted_y, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        multiplier = -0.05
        for i in range(10):
            print("epoch"+str(i))
            batch_size = 60
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                
                grads_W1, grads_b1, grads_W2, grads_b2, grads_W3, grads_b3, grads_W4, grads_b4, grads_W5, grads_b5 = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4, self.W5, self.b5])
                
                self.W1.update(grads_W1, multiplier)
                self.b1.update(grads_b1, multiplier)
                self.W2.update(grads_W2, multiplier)
                self.b2.update(grads_b2, multiplier)
                self.W3.update(grads_W3, multiplier)
                self.b3.update(grads_b3, multiplier)
                self.W4.update(grads_W4, multiplier)
                self.b4.update(grads_b4, multiplier)
                self.W5.update(grads_W5, multiplier)
                self.b5.update(grads_b5, multiplier)
            accuracy = dataset.get_validation_accuracy()
            print(accuracy)
            if accuracy > 0.985:
                    break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.n0 = self.num_chars
        self.n1 = 600
        self.n2 = 400
        self.n3 = 200
        self.ny = len(self.languages)
        self.W1 = nn.Parameter(self.n0, self.n1)
        self.W2 = nn.Parameter(self.n1, self.n2)
        self.W_hidden = nn.Parameter(self.ny, self.n1)
        self.b1 = nn.Parameter(1, self.n1)
        self.b2 = nn.Parameter(1, self.n2)
        self.b_hidden = nn.Parameter(1, self.n1)

        self.W3 = nn.Parameter(self.n2, self.n3)
        self.b3 = nn.Parameter(1, self.n3)
        self.W4 = nn.Parameter(self.n3, self.ny)
        self.b4 = nn.Parameter(1, self.ny)

        

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        i = 0
        for element in xs:
            Z1 = nn.AddBias(nn.Linear(element, self.W1), self.b1)
            if i == 0:
                A1 = nn.ReLU(Z1)
            else:
                Z_hidden = nn.AddBias(nn.Linear(h, self.W_hidden), self.b_hidden)
                A1 = nn.ReLU(nn.Add(Z_hidden, Z1))
            Z2 = nn.AddBias(nn.Linear(A1, self.W2), self.b2)
            A2 = nn.ReLU(Z2)
            Z3 = nn.AddBias(nn.Linear(A2, self.W3), self.b3)
            A3 = nn.ReLU(Z3)
            Z4 = nn.AddBias(nn.Linear(A3, self.W4), self.b4)
            i += 1
            h = nn.ReLU(Z4)
        return Z4
            

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(xs)
        loss = nn.SoftmaxLoss(predicted_y, y)
        return loss


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        multiplier = -0.01
        for i in range(1000):
            print("epoch"+str(i))
            batch_size = 50
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4, self.W_hidden, self.b_hidden])
                self.W1.update(grads[0], multiplier) 
                self.b1.update(grads[1], multiplier)
                self.W2.update(grads[2], multiplier)
                self.b2.update(grads[3], multiplier)
                self.W3.update(grads[4], multiplier) 
                self.b3.update(grads[5], multiplier)
                self.W4.update(grads[6], multiplier)
                self.b4.update(grads[7], multiplier)
                
                self.W_hidden.update(grads[8], multiplier) 
                self.b_hidden.update(grads[9], multiplier) 
            accuracy = dataset.get_validation_accuracy()
            print(accuracy)
            if accuracy > 0.83:
                    break

