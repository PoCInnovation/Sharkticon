# Sharkticon

## Description

Sharkticon is an intrusion detection system.

Its specificity is that it uses an anomaly detection system and machine learning, notably the transformers architecture.

Indeed, currently the most known IDS (intrusion detection system) use database of attack signatures to detect attacks.

Obviously a problem arises, if these systems face new attacks. This is not the case with our IDS, which is able to detect attacks that it has never seen thanks to anomaly detection.

See below a schema of the current architecture 

![Schema](./.github/assets/schema.png)

## Installation

```
 git clone https://github.com/PoCInnovation/Sharkticon.git
 cd Sharkticon
 pip3 install -r requirements.txt
```
## Quick Start

```
Sharkticon
or
Sharkticon --cli
```

If you use the CLI, you will have less information \
but the essentials like alerts will be available.

## Explanation

Sharkticon uses Wireshark to retrieve the network stream.
is then processed by a python script to render it in the format of our model.

For our model we use the transformers architecture, being the state of the art in NLP, we have adapted it and used it in our project. That's why we have focused on the HTTP protocol which is more verbose and therefore where the transformers exploits its qualities at best.

<p align="center">
    <br/>
  <img src="./.github/assets/transformers.png" />
  <br/>
  <br/>
</p>

Our model makes a prediction of the next packet from the previous ones, we then use our anomaly detection algorithm to detect if the packet is malicious, if X packets are malicious in a Y time frame then we raise an alert.

## Dependencies

|                          Dependency                        |      License       |
|:----------------------------------------------------------:|:------------------:|
| [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)              |  Apache License 2.0 |
| [numpy/numpy](https://github.com/numpy/numpy)      | 	BSD License        |


------------
## Maintainers

 - [Mikaël Vallenet](https://github.com/Mikatech)
 - [Evan Sabre](https://github.com/EvanSabre)