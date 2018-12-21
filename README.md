# dnn-control

This code demonstrates how to train optimal state-feedback controllers for a 2D quadcopter model represented by a feedforward neural network.

The properties of these controllers are investigated in the following papers:
* Tailor, D., Izzo, D. Learning the optimal state-feedback via supervised imitation learning. [In Preparation]
* [Izzo D., Tailor D., Vasileiou T. On the stability analysis of optimal state feedbacks as represented by deep neural models.]()

---

## Usage

This code has only been tested on a UNIX environment with Python 3.6.
We use `virtualenv` with `pip` for package management.

The code responsible for trajectory optimisation requires `ampl` and `snopt` executables to be in the system path.
Datasets are provided so the rest of the code is usable if these are unavailable.

The environment variable needs to be set: `PYTHONHASHSEED=0` as the code responsible for exporting and reading trained keras models depends on reproducible hashes.
It is easiest to add this to the *activate* script of virtualenv where the variable can be unset also on deactivation.
