const fs = require('fs');
const path = require('path');
const filePath = path.join(__dirname, 'file.txt');
async function readFile() {
    try {
        await fs.promises.access(filePath, fs.constants.F_OK);
        const data = await fs.promises.readFile(filePath, 'utf8');
        console.log(data);
    } catch (err) {
        console.error('Error reading the file:', err.message);
    }
}
readFile();
