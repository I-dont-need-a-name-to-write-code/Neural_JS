class NeuralNetwork {

    constructor(input_nodes = 0, hiddenLayerA_nodes = 0, hiddenLayerB_nodes = 0, 
                output_nodes = 0, learning_Rate = 0.2, activationFunc = "sigmoid") 
        {
        
        if(arguments[0] instanceof NeuralNetwork) {

            let brain = arguments[0];

            this.input_nodes = brain.input_nodes;
            this.hiddenLayerA_nodes = brain.hiddenLayerA_nodes;
            this.hiddenLayerB_nodes = brain.hiddenLayerB_nodes;
            this.output_nodes = brain.output_nodes;
            this.learning_Rate = brain.learning_Rate;
            this.activationFunc = brain.activationFunc;
            this.derivativeFunc = brain.derivativeFunc;

            this.weights_I_HA = brain.weights_I_HA;
            this.weights_HA_HB = brain.weights_HA_HB;
            this.weights_HB_O = brain.weights_HB_O;

            this.bias_HA = brain.bias_HA;
            this.bias_HB = brain.bias_HB;
            this.bias_O = brain.bias_O;
            
            this.functionPointer = brain.functionPointer;

            // this.error = brain.error;
        }

        else {

            this.input_nodes = input_nodes;
            this.hiddenLayerA_nodes = hiddenLayerA_nodes;
            this.hiddenLayerB_nodes = hiddenLayerB_nodes;
            this.output_nodes = output_nodes;
            this.learning_Rate = learning_Rate;

            this.functionPointer = 0;
            if(activationFunc == "sigmoid") {
                this.functionPointer = 1;
                this.activationFunc = sigmoid;
                this.derivativeFunc = dsigmoid;
            }
            if(activationFunc == "tanh") {
                this.functionPointer = 2;
                this.activationFunc = tanh;
                this.derivativeFunc = dtanh;
            } 
            if(activationFunc == "relu") {
                this.functionPointer = 3;
                this.activationFunc = Relu;
                this.derivativeFunc = dRelu;
            } 
            if(activationFunc == "leakyRelu") {
                this.functionPointer = 4;
                this.activationFunc = leakyRelu;
                this.derivativeFunc = dleakyRelu;
            } 


            this.error;

            // Create all weight matrices
            this.weights_I_HA = new Matrix(this.hiddenLayerA_nodes, this.input_nodes);
            this.weights_HA_HB = new Matrix(this.hiddenLayerB_nodes, this.hiddenLayerA_nodes);
            this.weights_HB_O = new Matrix(this.output_nodes, this.hiddenLayerB_nodes);

            this.weights_I_HA.randomize();
            this.weights_HA_HB.randomize();
            this.weights_HB_O.randomize();

            // Create all bias matrices
            this.bias_HA = new Matrix(this.hiddenLayerA_nodes, 1);
            this.bias_HB = new Matrix(this.hiddenLayerB_nodes, 1);
            this.bias_O = new Matrix(this.output_nodes, 1);

            this.bias_HA.randomize();
            this.bias_HB.randomize();
            this.bias_O.randomize();
            
        }
    }

    copy() {
        return new NeuralNetwork(this);
    }

    // feed forward algorithm
    predict(input_array) {
        let inputs = Matrix.fromArray(input_array);

        // Calculate hiddenA
        let hidden_A = Matrix.multiply(this.weights_I_HA, inputs);
        hidden_A.add(this.bias_HA);
        hidden_A.map(this.activationFunc);

        // Calculate hiddenB
        let hidden_B = Matrix.multiply(this.weights_HA_HB, hidden_A);
        hidden_B.add(this.bias_HB);
        hidden_B.map(this.activationFunc);

        // Calculate outputs
        let outputs = Matrix.multiply(this.weights_HB_O, hidden_B);
        outputs.add(this.bias_O);
        if(this.output_nodes > 1){ 
            outputs.softmax();
        }
        else {
            outputs.map(this.activationFunc);
        }

        return outputs.toArray();
    }


    setLearningRate(lr) {
        this.learning_Rate = lr;
    }

    // training algorithm for the neural network (Backpropagation algorithm)
    training(input_array, target_array) {

        let targets = Matrix.fromArray(target_array);

        // Recalculate all the layers
        let inputs = Matrix.fromArray(input_array);

        // Calculate hiddenA
        let hidden_A = Matrix.multiply(this.weights_I_HA, inputs);
        hidden_A.add(this.bias_HA);
        hidden_A.map(this.activationFunc);

        // Calculate hiddenB
        let hidden_B = Matrix.multiply(this.weights_HA_HB, hidden_A);
        hidden_B.add(this.bias_HB);
        hidden_B.map(this.activationFunc);

        // Calculate outputs
        let outputs = Matrix.multiply(this.weights_HB_O, hidden_B);
        outputs.add(this.bias_O);
        if(this.output_nodes > 1){ 
            outputs.softmax();
        }
        else {
            outputs.map(this.activationFunc);
        }


        // transpose all matrices

        let hiddenB_T = Matrix.transpose(hidden_B);
        let hiddenA_T = Matrix.transpose(hidden_A);
        let inputs_T = Matrix.transpose(inputs);

        // ----------------------------------------outputs------------------------------------------------

 
        // Calculate output errors and weights (HiddenB : output) deltas

        let output_error = Matrix.sub(targets, outputs);
        
        // this.error = new Matrix(output_error);

        // delta_weights_HB_O = leaningRate * output_error * derivativeOftheOutputs * hiddenB_T
        // delta_bias_HB_O = learningRate * output_error * derivativeOftheOutputs

        let gradient_HB_O = Matrix.map(outputs, this.derivativeFunc);
        gradient_HB_O.multiply(output_error);
        gradient_HB_O.multiply(this.learning_Rate);

        let delta_weights_HB_O = Matrix.multiply(gradient_HB_O, hiddenB_T);
        let delta_bias_HB_O = gradient_HB_O;


        // ----------------------------------------hiddenB------------------------------------------------
        // Calculate hiddenB errors

        let weights_HB_O_T = Matrix.transpose(this.weights_HB_O);


        let hiddenB_errors = Matrix.multiply(weights_HB_O_T, output_error);

        // delta_weights_HA_HB = leaningRate * hiddenB_errors * derivativeOftheHiddenB * hiddenA_T
        // delta_bias_HA_HB = learningRate * hiddenB_errors * derivativeOftheHiddenB
        let gradient_HA_HB = Matrix.map(hidden_B, this.derivativeFunc);
        gradient_HA_HB.multiply(hiddenB_errors);
        gradient_HA_HB.multiply(this.learning_Rate);

        let delta_weights_HA_HB = Matrix.multiply(gradient_HA_HB, hiddenA_T);
        let delta_bias_HA_HB = gradient_HA_HB;

        // ----------------------------------------hiddenA------------------------------------------------


        // Calculate hiddenA errors

        let weights_HA_HB_T = Matrix.transpose(this.weights_HA_HB);

        let hiddenA_errors = Matrix.multiply(weights_HA_HB_T, hiddenB_errors);

        // delta_weights_I_HA = leaningRate * hiddenA_errors * derivativeOftheHiddenA * inputs_T
        // delta_bias_I_HA = learningRate * hiddenA_errors * derivativeOftheHiddenA
        let gradient_I_HA = Matrix.map(hidden_A, this.derivativeFunc);
        gradient_I_HA.multiply(hiddenA_errors);
        gradient_I_HA.multiply(this.learning_Rate);

        let delta_weights_I_HA = Matrix.multiply(gradient_I_HA, inputs_T);
        let delta_bias_I_HA = gradient_I_HA;


        this.weights_HB_O.add(delta_weights_HB_O);
        this.bias_O.add(delta_bias_HB_O);
        this.weights_HA_HB.add(delta_weights_HA_HB);
        this.bias_HB.add(delta_bias_HA_HB);
        this.weights_I_HA.add(delta_weights_I_HA);
        this.bias_HA.add(delta_bias_I_HA);
    }


    // netError() { //not sure if this is correct but the error value seems to be decreasing with time
    //     let avgError = 0;
    //     for(let i = 0; i < this.error.rows; i++) {
    //         avgError += Math.abs(this.error.data[i][0]); 
    //     }
    //     avgError /= this.error.rows;
    //     return avgError;
    // }

    train(iterations , training_data) { 
        for(let i = 0; i < iterations ; i++) {
            let data = training_data[i];
            this.training(data.inputs, data.targets);
        }
    }

    trainRandom(iterations , training_data) { 
        for(let i = 0; i < iterations ; i++) {
            let data = random(training_data);
            this.training(data.inputs, data.targets);
        }
    }

    serialize() {
        return JSON.stringify(this);
    }

    static deserialize(data) {

        if(typeof data == 'string') {
            data = JSON.parse(data);
        }
        
        let newBrain = new NeuralNetwork(data.input_nodes, data.hiddenLayerA_nodes,
                                         data.hiddenLayerB_nodes, data.output_nodes, data.learning_Rate);

        if(data.functionPointer == 1) {
            newBrain.activationFunc = sigmoid;
            newBrain.derivativeFunc = dsigmoid;
        }
        if(data.functionPointer == 2) {
            newBrain.activationFunc = tanh;
            newBrain.derivativeFunc = dtanh;
        } 
        if(data.functionPointer == 3) {
            newBrain.activationFunc = Relu;
            newBrain.derivativeFunc = dRelu;
        } 
        if(data.functionPointer == 4) {
            newBrain.activationFunc = leakyRelu;
            newBrain.derivativeFunc = dleakyRelu;
        } 
        newBrain.weights_I_HA = Matrix.deserialize(data.weights_I_HA);
        newBrain.weights_HA_HB = Matrix.deserialize(data.weights_HA_HB);
        newBrain.weights_HB_O = Matrix.deserialize(data.weights_HB_O);

        newBrain.bias_HA = Matrix.deserialize(data.bias_HA);
        newBrain.bias_HB = Matrix.deserialize(data.bias_HB);
        newBrain.bias_O = Matrix.deserialize(data.bias_O);

        return newBrain;
    }

}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function dsigmoid(x) { 
    return x * (1 - x);
}

function tanh(x) {
    return Math.tanh(x);
}

function dtanh(x) {
    return 1 - (x * x);
}

function Relu(x) {
    return Math.max(0, x);
}

function dRelu(y) {
    return (y > 0);
}

function leakyRelu(x) {
    return Math.max((0.001 * x) , x);
}

function dleakyRelu(y) {
    if(y > 0) {
        return 1;
    }
    else {
        return 0.001;
    }
}

function argMax(array) {
    return array.indexOf(max(array)); 
}
