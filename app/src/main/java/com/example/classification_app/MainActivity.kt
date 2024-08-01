package com.example.classification_app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.rememberImagePainter
import com.example.classification_app.ui.theme.Classification_appTheme
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : ComponentActivity() {

    private lateinit var interpreter: Interpreter
    private var labels = mutableListOf<String>()
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            Classification_appTheme {
                MyApp()
            }
        }

        loadModel()
        loadLabels()
    }

    @Composable
    fun MyApp(
        modifier: Modifier = Modifier
    ) {
        var selectedImages by remember { mutableStateOf<List<Uri>>(emptyList()) }
        var predictions by remember { mutableStateOf<List<String>>(emptyList()) }
        var isSelectButtonEnabled by remember { mutableStateOf(true) }
        var isPredictButtonEnabled by remember { mutableStateOf(false) }

        val launcher = rememberLauncherForActivityResult(contract = ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                selectedImages = selectedImages + it
            }
        }

        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            Column(
                modifier = modifier
                    .padding(vertical = 15.dp, horizontal = 4.dp),
                verticalArrangement = Arrangement.Top,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(
                    text = "Classification",
                    fontSize = 18.sp
                )

                Spacer(modifier = Modifier.height(12.dp))

                Box(
                    modifier = Modifier.size(width = 300.dp, height = 320.dp)
                ) {
                    LazyColumn(
                        modifier = Modifier
                            .padding(12.dp)
                            .fillMaxSize(),
                        verticalArrangement = Arrangement.Top,
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        itemsIndexed(selectedImages) { index, uri ->
                            val painter = rememberImagePainter(uri)
                            Image(
                                painter = painter,
                                contentDescription = null,
                                modifier = Modifier.size(150.dp)
                            )
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(text = "Prediction : ${if(predictions.isNotEmpty()) predictions[index] else "NA"}")
                        }
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                Button(
                    onClick = {
                        launcher.launch("image/*")
                    },
                    modifier = Modifier.size(width = 150.dp, height = 40.dp),
                    enabled = isSelectButtonEnabled,
                ) {
                    Text(text = "Select Images")
                }

                Spacer(modifier = Modifier.height(12.dp))

                Button(
                    onClick = {
                        selectedImages = emptyList()
                        predictions = emptyList()
                        isSelectButtonEnabled = true
                        isPredictButtonEnabled = false
                    },
                    modifier = Modifier.size(width = 150.dp, height = 40.dp),
                ) {
                    Text(text = "Remove Images")
                }

                Spacer(modifier = Modifier.height(12.dp))

                if(selectedImages.isNotEmpty()) {
                    isPredictButtonEnabled = true
                }

                Button(
                    onClick = {
                        predictions = emptyList()
                        isSelectButtonEnabled = false
                        for (image in selectedImages) {
                            val inputImageBuffer = processImage(image)

                            val outputData = runInference(inputImageBuffer)
                            val predictedClass = outputData.withIndex().maxByOrNull { it.value }?.index ?: 0

                            predictions = predictions + labels[predictedClass]
                        }
                    },
                    modifier = Modifier.size(width = 150.dp, height = 40.dp),
                    enabled = isPredictButtonEnabled,
                ) {
                    Text(text = "Get Predictions")
                }
            }
        }
    }

    private fun loadModel() {
        try {
            val inputStream = assets.open("mobilenet_v1_1.0_224_quant.tflite")
            val modelBytes = inputStream.readBytes()
            val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size)
            modelBuffer.order(ByteOrder.nativeOrder())
            modelBuffer.put(modelBytes)
            modelBuffer.rewind()

            interpreter = Interpreter(modelBuffer)
            inputStream.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun loadLabels() {
        try {
            val inputStream = assets.open("labels.txt")
            inputStream.bufferedReader().useLines { lines ->
                lines.forEach {
                    labels.add(it)
                }
            }
            inputStream.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun processImage(image: Uri): ByteBuffer {
        val inputStream = contentResolver.openInputStream(image)
        val bitmap = BitmapFactory.decodeStream(inputStream)

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        val byteBuffer = ByteBuffer.allocateDirect(224 * 224 * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val px = resizedBitmap.getPixel(x, y)

                byteBuffer.put((px shr 16 and 0xFF).toByte())  // Red channel
                byteBuffer.put((px shr 8 and 0xFF).toByte())   // Green channel
                byteBuffer.put((px and 0xFF).toByte())         // Blue channel
            }
        }

        byteBuffer.rewind()
        return byteBuffer
    }

    private fun runInference(inputImage: ByteBuffer): FloatArray {
        val outputArray = Array(1) { ByteArray(1001) }
        interpreter.run(inputImage, outputArray)

        val floatOutputArray = FloatArray(outputArray[0].size)
        for (i in outputArray[0].indices) {
            floatOutputArray[i] = (outputArray[0][i].toInt() and 0xFF) * 0.00390625f
        }

        return floatOutputArray
    }

    @Preview(showBackground = true)
    @Composable
    fun ClassificationPreview() {
        Classification_appTheme {
            MyApp()
        }
    }
}