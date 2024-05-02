## Task 5 Readme

### Abstract

I've modified the previous ising and defuant model by applying the small-world model on them instead of using numpy array. And then I updated the section that processing the system arguments with a new commands `-use_network <N>`. After that I created a function to generate a animation to display the evolution of a given network sequence.

#### Modification on Ising model

In this section I mainly modified the previous function `calculate_agreement` and `ising_step`. I've combined them up with a small-world model.

You can now call an ising model using network by this function `ising_network(population, alpha=None, external=0.0)`

- population, an integer that indicate the population of the network.
- alpha, the parameter to compute the probability of flipping the agreement.
- external, a paramter that decides the effect of an external pull on opinions.

This function will return a small-world model network after applying a given ising model on it.

#### Modification on Defuant model

In this section I mainly modified the previous function `defuant_model`. I've applied a small-world model on it.

You can now call a defuant model using network by this function `defuant_network(population, threshold, beta, iterations)`

- population, an integer that indicate the population of the network.
- threshold, a float number that decides the threshold for interaction.
- beta, a float number that decides the "velocity" of updating.
- iterations, an integer that indicates the total iterations we need.

This function will return a small-world model network after applying a given ising model on it.

#### Modification on Input Arguments Acception

In this section I mainly modified the input processing of task-1 and task-2. I've added a filter of argument `-use_network`. If this was detected while processing task-1 and task-2 input, the model will using the network to solve the problem.

You can now call an ising model using network by this command, `<N>` here refers to the population:

```
python3 FCP_assignment_v3.py -ising_model -use_network <N>
```

And you can now call an defuant model using network by this command, `<N>` here refers to the population:

```
python3 FCP_assignment_v3.py -defuant -use_network <N>
```

#### Animation Generating

In this section I mainly add two functions `NetPlot(fig, ax, net, current_frame)` and `animationNet(nets_seq)` to generate an animation for a net-work processing.

The only argument here you may care is `nets_seq`. It's a sequence of networks. In fact, it's a list of several networks. It will then show each network in the sequence and display the current frame index in the title.

This function has been called through the ising model and defuant model while using network. It will show you the last frame of the animation and save the animation as "animation.gif" in your current path.
