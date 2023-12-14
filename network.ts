type Activation = {
  derivative: (input: number) => number;
  activate: (input: number) => number;
};

const linearActivation: Activation = {
  derivative: (_input: number) => {
    return 1;
  },
  activate: (input: number) => {
    return input;
  },
};

const reluActivation: Activation = {
  derivative: (input: number) => {
    return input > 0 ? 1 : 0;
  },
  activate: (input: number) => {
    return Math.max(0, input);
  },
};

class Neuron {
  weights: number[];
  bias: number;
  activation: Activation;
  weightedSumWithBias: number = 0;
  input: number[] = [];
  dInput: number[] = [];
  dWeights: number[] = [];
  dBias: number = 0;
  learningRate = 0.001;

  constructor({ nInputs, activation }: { nInputs: number; activation: Activation }) {
    // todo -> replace Math.random() with something more appropriate
    this.weights = [...Array(nInputs)].map((_) => Math.random() * 0.01);
    this.bias = Math.random() * 0.01;
    this.activation = activation;
  }

  forward({ input }: { input: number[] }): number {
    this.input = input;
    const weightedSumWithBias = input.reduce(
      (sum, input, index) => sum + input * this.weights[index],
      this.bias
    );
    this.weightedSumWithBias = weightedSumWithBias;

    return this.activation.activate(weightedSumWithBias);
  }

  backward(gradient: number) {
    // Calculate gradients with respect to weights and bias
    this.dWeights = this.weights.map(
      (_, index) =>
        gradient *
        this.activation.derivative(this.weightedSumWithBias) *
        this.input[index]
    );
    this.dInput = this.input.map(
      (_, index) =>
        gradient *
        this.activation.derivative(this.weightedSumWithBias) *
        this.weights[index]
    );
    this.dBias = gradient * this.activation.derivative(this.weightedSumWithBias);

    // Update weights and bias
    this.weights = this.weights.map(
      (weight, index) => weight - this.learningRate * this.dWeights[index]
    );
    this.bias = this.bias - this.learningRate * this.dBias;
  }
}

class DenseLayer {
  neurons: Neuron[] = [];
  dInputs: number[] = [];

  constructor({
    nInputs,
    nNeurons,
    activation,
  }: {
    nInputs: number;
    nNeurons: number;
    activation: Activation;
  }) {
    this.neurons = [...Array(nNeurons)].map((_) => new Neuron({ nInputs, activation }));
    this.dInputs = new Array(nInputs).fill(0);
  }

  forward({ input }: { input: number[] }): number[] {
    return this.neurons.map((n) => n.forward({ input }));
  }

  backward(gradient: number[]) {
    // Reset dInputs before accumulating for each neuron
    this.dInputs = new Array(this.neurons[0].input.length).fill(0);

    // Calculate dInputs for each neuron in the layer
    for (let i = 0; i < this.neurons.length; i++) {
      const neuron = this.neurons[i];
      neuron.backward(gradient[i]);

      for (let j = 0; j < this.dInputs.length; j++) {
        this.dInputs[j] += neuron.dInput[j];
      }
    }
  }
}

class NeuralNetwork {
  layers: DenseLayer[] = [];

  constructor({ layers }: { layers: DenseLayer[] }) {
    this.layers = layers;
  }

  forward({ input }: { input: number[] }): number[] {
    let data = input;

    this.layers.forEach((layer) => {
      data = layer.forward({ input: data });
    });

    return data;
  }

  backward(expectedOutput: number[], actualOutput: number[]) {
    let gradient = expectedOutput.map(
      (expected, index) => 2 * (actualOutput[index] - expected)
    );

    for (let i = this.layers.length - 1; i >= 0; i--) {
      this.layers[i].backward(gradient);
      gradient = [...this.layers[i].dInputs];
    }
  }

  totalLoss(expectedOutput: number[], actualOutput: number[]): number {
    return expectedOutput.reduce(
      (sum, expected, index) => sum + Math.pow(expected - actualOutput[index], 2),
      0
    );
  }

  loss(expectedOutput: number[], actualOutput: number[]): number[] {
    return expectedOutput.map((expected, index) =>
      Math.pow(expected - actualOutput[index], 2)
    );
  }
}

const network = new NeuralNetwork({
  layers: [
    new DenseLayer({
      nInputs: 1,
      nNeurons: 3,
      activation: reluActivation,
    }),
    new DenseLayer({
      nInputs: 3,
      nNeurons: 1,
      activation: linearActivation,
    }),
  ],
});

// given dataset
let dataset = [
  [[0], [2]],
  [[1], [3]],
  [[2], [4]],
  [[3], [5]],
  [[4], [6]],
  [[5], [7]],
  [[6], [8]],
  [[7], [9]],
  [[8], [10]],
  [[9], [11]],
];

// train network
for (let i = 0; i < 500; i++) {
  for (let j = 0; j < dataset.length; j++) {
    const actualOutput = network.forward({ input: dataset[j][0] });
    network.backward(dataset[j][1], actualOutput);

    if (i % 100 === 0) {
      const loss = network.loss(dataset[j][1], actualOutput);
      const totalLoss = network.totalLoss(dataset[j][1], actualOutput);
      console.log("total loss:", totalLoss);
      console.log("loss:", loss);
    }
  }
}

// given test data
let testData = [
  [[-19], [-17]],
  [[12], [14]],
  [[13], [15]],
  [[14], [16]],
  [[15], [17]],
  [[123], [125]],
];

// test network
for (let i = 0; i < testData.length; i++) {
  const actual = network.forward({ input: testData[i][0] });
  console.log("expected", testData[i][1]);
  console.log("actual", actual);
}
