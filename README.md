# Data Analyst R

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ariellubonja/data-analyst-r.git
cd data-analyst-r
```

### 2. Install the Required Python Packages

If a `requirements.txt` file is provided, install dependencies with:

```bash
pip install -r requirements.txt
```

Otherwise, manually install the necessary packages:

```bash
pip install rpy2 openai autogen_core autogen_ext autogen_agentchat
```

### 3. Set Your OpenAI API Key

On macOS/Linux:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

On Windows (Command Prompt):

```cmd
set OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### `human_in_the_loop_rpy2.py`

This script demonstrates an interactive agent system using `rpy2` for in-process R code execution.

#### **What It Does:**
- **Assistant Agent**: Generates R code wrapped in a triple backtick code fence (```r ... ```).
- **Executor Agent**: Extracts and executes the R code using `rpy2`, capturing console output and errors.
- **Human Feedback Loop**: Allows corrections or new instructions until the analysis is complete.

#### **How to Run:**
```bash
python human_in_the_loop_rpy2.py
```

The script starts with an initial task, e.g.:

```
Compute mean and standard deviation from the 'aorta' column in dsc.csv. Then say 'DONE'.
```

You can provide additional feedback or corrections after each cycle. Press **Enter** on an empty input to exit.

---

### `single_file_agent.py`

This script implements a self-contained, iterative chat system using a **Round Robin** architecture.

#### **What It Does:**
- **RCoderAgent**: Generates and refines R code iteratively based on feedback.
- **RExecutorAgent**: Identifies R code in messages and executes it using the system’s `Rscript` command via a temporary file.
- **Termination Conditions**: The system stops when:
  - `RCoderAgent` outputs `"DONE"`, or
  - A set number of message exchanges is reached.

#### **How to Run:**
```bash
python single_file_agent.py
```

The agents will interact, generating and executing R code until the task is successfully completed.

---

## Project Structure

```
data-analyst-r/
├── human_in_the_loop_rpy2.py    # Interactive agent system using rpy2 for R execution with human feedback.
├── single_file_agent.py         # Self-contained agent chat using subprocess to execute R via Rscript.
├── README.md                    # This documentation file.
└── requirements.txt             # (Optional) List of Python package dependencies.
```

---

## Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository.
2. **Create a new branch** for your feature or fix.
3. **Submit a pull request**, detailing your changes.

Feel free to open issues for bug reports or feature requests.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Acknowledgments

- **Inspiration**: Based on the concept of ChatGPT's Data Analyst.
- **Frameworks**: Built using the `autogen` framework for multi-agent interaction.
- **Tools**: Thanks to the developers of `rpy2`, OpenAI, and the broader R and Python communities.

---

This README provides a clear and structured overview of the repository, usage instructions, and contribution guidelines. Enjoy exploring and extending your R data analyst assistant! 🚀
