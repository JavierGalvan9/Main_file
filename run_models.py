
import itertools
import subprocess

## Run all the bashCommands with all the possible combinations of parameters
# Define the parameters of the python program
parameters = {
    'oversampling': ['oversampling', 'no-oversampling'],
    'undersampling': ['undersampling', 'no-undersampling'],
    'cost_sensitive': ['cost_sensitive', 'no-cost_sensitive'],
    'dropout': ['dropout', 'no-dropout'],
    'L2_regularizer': ['L2_regularizer', 'no-L2_regularizer']
}
# Get all the possible combinations of parameters
combinations = list(itertools.product(*parameters.values()))
# Run all the bashCommands with all the possible combinations of parameters
for combination in combinations:
    # Create string with all the items of the combination
    combination_string = "_".join(combination)
    # Create the bashCommand which contains a string of the python program inside a string of the bash command
    # and the parameters of the python program
    pythonCommand = f"python xylella_detection_nn.py"
    for i in range(len(combination)):
        pythonCommand += " --" + str(combination[i])
    # Create the output and error files
    out_error_Command = f"touch Out/{combination_string}.out Error/{combination_string}.err"
    subprocess.run(out_error_Command, shell=True, check=True)
    # Create the bashCommand which contains the pythonCommand inside a string of the bash command
    bashCommand = f"/usr/local/bin/run -c 1 -m 5 -j {combination_string} -t 20:30 -o Out/{combination_string}.out -e Error/{combination_string}.err '{pythonCommand}'"
    subprocess.run(bashCommand, shell=True, check=True)

