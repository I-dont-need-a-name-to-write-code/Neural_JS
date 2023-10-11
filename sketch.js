let brain;
let iterCounter = 0;
let iterLimit   = 1000;

let totalImagesTraversed = 0;
let imagesGuessedCorrectly = 0;

let iterTest = 0;
let iterTestLimit = 5000;
let flag  = false;

function setup() {
    createCanvas(280,280);
    background(0);
  //brain = new NeuralNetwork(784, 64, 32, 10, 0.001, "leakyRelu");
    brain = NeuralNetwork.deserialize(model_data);
    console.log("Starting Training !!!");
}

function draw() {

    // Test
    if(flag) {
        let current_test_data = random(test_data);
        
        showImage(current_test_data.inputs, 28, 28);
        let prediction = brain.predict(current_test_data.inputs);
        let target = argMax(current_test_data.targets);
        let guess = argMax(prediction);
        console.log("Target : " + target);
        console.log("Prediction : " + guess);
        
        if(target == guess) { ++imagesGuessedCorrectly; }
        
        console.log("Testing Iter : " + ++iterTest);
        console.log("Number of Images Tested : " + (++totalImagesTraversed));
        console.log("Accuracy : " + (imagesGuessedCorrectly / totalImagesTraversed) * 100);

        if(iterTest == iterTestLimit) {
          noLoop();
        }
        
    } else {
        // train
        brain.trainRandom(1000, training_data); 
        let randomData = random(training_data);
        let prediction = brain.predict(randomData.inputs);
        let target = randomData.targets;
        console.table(prediction);
        console.table("Prediction : " + argMax(prediction));
        console.table("Target : " + argMax(target));
        
        // Show Image
        showImage(randomData.inputs, 28, 28);
        
        console.log("Iter : " + (++iterCounter));
        if(iterCounter == iterLimit) {
          console.log("Training Complete !!!");
          console.log("Testing the NN now !!!");
          flag = true;
        }
    }

}

function loadUserInputs() {
    let inputs = [];
    let img = get();
    img.resize(28,28);
    img.loadPixels();
    for(let i = 0; i < img.pixels.length; i++) {
      if(i % 4 == 0)
        inputs.push(img.pixels[i] / 255);
    }
    img.updatePixels();
    return inputs;
}


function showImage(array , width, height) {
    let img = createImage(width, height);
    img.loadPixels();
    for(let i = 0; i < array.length; i++) {
      val = array[i] * 255.0;
      img.pixels[i * 4 + 0] = val;
      img.pixels[i * 4 + 1] = val;
      img.pixels[i * 4 + 2] = val;
      img.pixels[i * 4 + 3] = 255;
    }
    img.updatePixels();
    img.resize(280,280);
    image(img, 0, 0);
}



