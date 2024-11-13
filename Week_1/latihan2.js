//asinkron karena kita memuat data csv dari url yang membutuhkan waktu(asinkron)
async function run() {
    const csvUrl="/data/iris.csv";
    //Memuat Data CSV ke dalam TensorFlow.js
    const trainingData = tf.data.csv(csvUrl, {
        columnConfigs:{ // konfigurasi untuk kolom-kolom dalam file CSV
            //menandai kolom spesies adalah label
            species: {
                isLabel : true
            }
        }
    }) ;
    
    const numOfFeatures = (await trainingData.columnNames()).length - 1;
    const numOfSamples = 150; //karena ada 150 baris sample data
    
    //Mengonversi Data CSV ke Format yang Dapat Digunakan
    const convertedData = 
    //mau mengubah jadi array
    //konvesri entri tiap dataset trainingData menjadi bentuk yang sesuai
     trainingData.map(({xs, ys})=>{ //dictionary, dan mengatakan ingin xs,ys bentuknya seperti ini
        const labels=[
            //mengubah string dalam label menjadi one-hot encoding ([1,0,0],[0,1,0],dll)
            ys.species == "setosa" ? 1:0, //1,0,0 posisinya
            ys.species == "virginica" ? 1:0, //0,1,0
            ys.species == "versicolor" ? 1:0 //0,0,1
        ]

        return { 
            xs: Object.values(xs), ys:Object.values(labels)};

            //xs diammbil semua nilainya dan dikonvesi jadi array ex. xs: [5.1, 3.5, 1.4, 0.2],
            //konversi ys(label) menjadi array ex. ys: [1, 0, 0]
       
     }).batch(10);



const model = tf.sequential();

model.add(tf.layers.dense({ //fully connected layer
    inputShape: [numOfFeatures], //ada 4 fitur //tidak ada flatten diisni
    activation: "sigmoid", 
    units: 5})); //nilai antara 0-1, nggak papa kalau di hidden layer
//output layer
model.add(tf.layers.dense({activation:"softmax", units:3}));

model.compile({
    loss:"categoricalCrossentropy",
    optimizer: tf.train.adam(0.06)}
);

//Train 
//await digunakan untuk menunggu hingga pelatihan selesai sebelum mengeksekusi kode berikutnya.
await model.fitDataset(
    convertedData, //latih menggunakan dataset yang telah dikonversi 
    {
        epochs:100,
        callbacks: { //melacak kemajuan pelatihan
            onEpochEnd:  async(epoch, logs) =>{ 
                console.log("E: " + epoch+"Loss: "+ logs.loss); //menampilkan nilai loss
            }
        }
    });

//inferensi model (Test)
// Setosa
const testVal = tf.tensor2d([4.4, 2.9, 1.4, 0.2], [1, 4]);
            
// Versicolor
// const testVal = tf.tensor2d([6.4, 3.2, 4.5, 1.5], [1, 4]);
            
// Virginica
// const testVal = tf.tensor2d([5.8,2.7,5.1,1.9], [1, 4]);

const prediction = model.predict(testVal);  //prediksi kelas dengan model yang telah dibuat
// alert(prediction); //akan menampilkan 3 kemungkinan /3 kelas
const pIndex = tf.argMax(prediction, axis=1).dataSync(); //mencari index dengan nilai probabilitas/prediksi tertinggi
const classNames = ["Setosa", "Virginica", "Versicolor"];
alert(classNames[pIndex])
}

run();
