//MNIST Data Set : Data set yang terkenal dalam tugas compvi seperti mengenalan karakter (huruf) atau digit

model=tf.sequential()

model.add(tf.layers.conv2d({
    inputShape: [28,28,1], //input gambar 28x28, grayscale
    kernelSize: 3, //filter blok 3x3 berjalan diatas gambar (proses 9 pixexl)
    filters: 8, //menghasikan 8 fitur maps
    activation:'relu' //non-negative
}));

model.add(tf.layers.maxPooling2d({poolSize:[2,2]}));

model.add(tf.layers.conv2d({
    filters: 16,
    kernelSize: 3,
    activation:'relu'
}));

model.add(tf.layers.maxPooling2d({poolSize:[2,2]}));

model.add(tf.layers.fatten())

model.add(tf.layers.dense({
    units:128,
    activation:'relu'
}));

model.add(tf.layers.dense({
    units:10,
    activation:'softmax'
}))

//compile model
model.compile(
    {
        optimizer: tf.train.adam(),
        loss: "categoricalCrossentropy",
        metrics:['accuracy']
    }
)

const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
const container = {name: 'Model Training', styles: {height:'1000px'}};
const fitCallbacks=tfvis.show.fitCallbacks(container, metrics)

//train model
model.fit(trainXs, trainYs, {
    batchSize: 20,
    validationData:[testXs, testYs],
    epochs: 20,
    shuffle:true,
    callbacks:fitCallbacks
})

//tf-vis library










