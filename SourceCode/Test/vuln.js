const fs = require('fs');
const path = require('path');
fs.readFile(path.join(__dirname, '../..', 'config.txt'), (err, data) => {
    if (err) {
        console.error(err);
    } else {
        console.log(data.toString());
    }
});