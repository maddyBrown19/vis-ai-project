import * as tf from '@tensorflow/tfjs';
import {IMAGE_H, IMAGE_W} from './data';

export const createConvModel = () => {
    // Create a sequential neural network model. tf.sequential provides an API
    // for creating "stacked" models where the output from one layer is used as
    // the input to the next layer.
    const model = tf.sequential();
  
    // The first layer of the convolutional neural network plays a dual role:
    // it is both the input layer of the neural network and a layer that performs
    // the first convolution operation on the input. It receives the 28x28 pixels
    // black and white images. This input layer uses 16 filters with a kernel size
    // of 5 pixels each. It uses a simple RELU activation function which pretty
    // much just looks like this: __/
    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_H, IMAGE_W, 1],
      kernelSize: 3,
      filters: 16,
      activation: 'relu'
    }));
  
    // After the first layer we include a MaxPooling layer. This acts as a sort of
    // downsampling using max values in a region instead of averaging.
    // https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  
    // Our third layer is another convolution, this time with 32 filters.
    model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
  
    // Max pooling again.
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  
    // Add another conv2d layer.
    model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
  
    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten({}));
  
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));
  
    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
    // represent numbers, but it's the same idea if you had classes that
    // represented other entities like dogs and cats (two output classes: 0, 1).
    // We use the softmax function as the activation for the output layer as it
    // creates a probability distribution over our 10 classes so their output
    // values sum to 1.
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  
    return model;
  }


  /**
 * Compile and train the given model.
 *
 * @param {tf.Model} model The model to train.
 * @param {onIterationCallback} onIteration A callback to execute every 10
 *     batches & epoch end.
 */
  export async function train(model, data, onIteration) {
  
    const optimizer = 'rmsprop';
    const trainData = data.getTrainData();
    const testData = data.getTestData();

    model.compile({
      optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
    // Batch size is another important hyperparameter. It defines the number of
    // examples we group together, or batch, between updates to the model's
    // weights during training. A value that is too low will update weights using
    // too few examples and will not generalize well. Larger batch sizes require
    // more memory resources and aren't guaranteed to perform better.
    const batchSize = 320;
  
    // Leave out the last 15% of the training data for validation, to monitor
    // overfitting during training.
    const validationSplit = 0.15;
  
    // Get number of training epochs from the UI.
    const trainEpochs = 3;
  
    // We'll keep a buffer of loss and accuracy values over time.
    let trainBatchCount = 0;

  
    const totalNumBatches =
        Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) *
        trainEpochs;
  
    // During the long-running fit() call for model training, we include
    // callbacks, so that we can plot the loss and accuracy values in the page
    // as the training progresses.
    let valAcc;
    await model.fit(trainData.xs, trainData.labels, {
      batchSize,
      validationSplit,
      epochs: trainEpochs,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          trainBatchCount++;
          // ui.logStatus(
          //     `Training... (` +
          //     `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` +
          //     ` complete). To stop training, refresh or close page.`);
          // ui.plotLoss(trainBatchCount, logs.loss, 'train');
          // ui.plotAccuracy(trainBatchCount, logs.acc, 'train');
          if (onIteration && batch % 10 === 0) {
            onIteration('onBatchEnd', trainBatchCount, logs);
          }
          await tf.nextFrame();
        },
        onEpochEnd: async (epoch, logs) => {
          valAcc = logs.val_acc;
          // ui.plotLoss(trainBatchCount, logs.val_loss, 'validation');
          // ui.plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
          if (onIteration) {
            onIteration('onEpochEnd', epoch, logs);
          }
          await tf.nextFrame();
        }
      }
    });
  
    const testResult = model.evaluate(testData.xs, testData.labels);
    const testAccPercent = testResult[1].dataSync()[0] * 100;
    const finalValAccPercent = valAcc * 100;
    console.info(
        `Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
        `Final test accuracy: ${testAccPercent.toFixed(1)}%`);
  }
  