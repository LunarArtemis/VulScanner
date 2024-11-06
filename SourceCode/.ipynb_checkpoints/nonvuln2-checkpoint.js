// Function to calculate the factorial of a number
function factorial(n) {
    if (n < 0) {
        return "Factorial is not defined for negative numbers";
    } else if (n === 0 || n === 1) {
        return 1; // Base case
    } else {
        return n * factorial(n - 1); // Recursive case
    }
}

// Example usage
const number = 5;
const result = factorial(number);
console.log(`The factorial of ${number} is ${result}`);
