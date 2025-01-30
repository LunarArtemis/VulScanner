const fileName = req.query.file;
const filePath = path.join(__dirname, fileName);
fs.readFile(filePath, 'utf8', (err, data) => {  
});