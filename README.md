# Multi-GPT-LLM: A Multi-Modal Large Language Model

This project introduces a multimodal large language model aimed at narrowing the gap between AI perception and human cognition, moving towards Artificial General Intelligence (AGI). By incorporating text, audio, and image data, this model enhances the ability to perform various tasks, such as image description, visual question answering, and image manipulation. Our focus includes multiple languages, with a particular emphasis on Arabic.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project seeks to build an AI system capable of human-like interactions by processing and synthesizing information from text, audio, and visual inputs. It includes support for multiple languages and focuses on bridging the perceptual gap between current AI and human cognition.

Key Features:
- Multimodal interaction (text, image, and audio input)
- Support for multiple languages (with a focus on Arabic)
- Advanced tasks like visual question answering, image captioning, and more

## Features
- **Text + Image to Image**: Generate images based on text and visual input.
- **Text + Image to Text**: Generate descriptive text based on a combination of text and visual input.
- **Audio to Text**: Convert speech into text with advanced speech recognition.

For more detailed feature information, visit the [Features](./docs/features.md) section.

## Installation
### 1. Clone the Repository
``` bash
git clone https://github.com/Mohamed-Samy26/Multi-GPT-LLM.git
cd Multi-GPT-LLM
```
### 2. Set Up the Environment
``` bash
conda env create -f environment.yml
conda activate MM-LLM
```

### 3. Prepare Pretrained Models
Download the pretrained LLM weights and set the corresponding paths in the config files.

For more details, see the [Installation Guide](./docs/installation.md).

## Usage
To run the project, follow the steps outlined in the [User Guide](./docs/user_guide.md). It contains detailed information on how to input text, audio, and image prompts.

``` bash
python app.py
```
## Contributing
Contributions are welcome! Please see the [Contributing Guidelines](./docs/contributing.md) for more information on how to get involved.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.
