package com.example.pythonexperiment;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    TextView textView;
    ImageView imageView;
    PyObject pyf;


    public void  procImg(View view){
        textView.setText("loading");
        Log.i("checking","loading");
        Bitmap b= BitmapFactory.decodeResource(getResources(), R.drawable.test);

        b = b.copy( Bitmap.Config.ARGB_8888 , true);
        Bitmap scaled = Bitmap.createScaledBitmap(b, 500, 800, true);
        int w = b.getWidth();
        int h = b.getHeight();
        int colour;
        int[][] red = new int[500][800];
        int[][] green = new int[500][800];
        int[][] blue=new int[500][800];
        int[][] alpha=new int[500][800];

        for(int i=0;i<500;i++)
            for(int j=0;j<800;j++)
            {
                colour=scaled.getPixel(i,j);
                red[i][j]=Color.red(colour);
                green[i][j]=Color.green(colour);
                blue[i][j]=Color.blue(colour);
                alpha[i][j]= Color.alpha(colour);
            }
        int[][][] data = pyf.callAttr("test", red,green,blue).toJava(int[][][].class);
        for(int x=0; x<500;x++) {
            for(int y=0; y<800; y++) {

                int color = Color.argb(255,data[0][y][x], data[1][y][x], data[2][y][x]);

                scaled.setPixel(x, y, color);
            }
        }
        int[] intArray = new int[]{ w,h };
        Bitmap scaledup = Bitmap.createScaledBitmap(scaled, 625, 1000, true);
        imageView.setImageBitmap(scaledup);
        textView.setText(Arrays.toString(intArray));
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textView=(TextView)findViewById(R.id.textView);
        imageView=(ImageView)findViewById(R.id.imageView);
        imageView.setImageResource(R.drawable.test);

        if (! Python.isStarted())
            Python.start(new AndroidPlatform(this));

        Python py=Python.getInstance();
        pyf =py.getModule("hello");

    }
}
