# Neurons: A simple neural network implementation

This repository contains a basic implementation of a neural network in TypeScript. The neural network is designed with modularity in mind, allowing customization of activation functions, layer sizes, and learning parameters.

## Usage

1. Given dataset

```typescript
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
```

2. Initialize the Neural Network:

```typescript
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
```

3. Train network

```typescript
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
```

4. Given test data

```typescript
let testData = [
  [[-19], [-17]],
  [[12], [14]],
  [[13], [15]],
  [[14], [16]],
  [[15], [17]],
  [[123], [125]],
];
```

5. Test network

```typescript
for (let i = 0; i < testData.length; i++) {
  const actual = network.forward({ input: testData[i][0] });
  console.log("expected", testData[i][1]);
  console.log("actual", actual);
}
```
