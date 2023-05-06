




# Define a function to calculate the average
def calculate_average(numbers):
    total = sum(numbers)
    average = total / len(numbers)
    return average


if __name__ == '__main__':

    
    # Ask the user to input a list of numbers separated by spaces
    number_list = input("Enter a list of numbers separated by spaces: ")

    # Convert the input string into a list of floats
    numbers = [float(num) for num in number_list.split()]

    # Calculate the average and print the result
    average = calculate_average(numbers)
    print("The average is:", average)

