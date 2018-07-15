package com.amazonaws.ml.mms.metrics;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.File;

public class MetricCollector {

        public StringBuilder stringBuilder;
        public String jsonString;

        public MetricCollector() {
            StringBuilder stringBuilder = null;
            String jsonString = null;
        }
        public void collect() throws IOException {

            this.stringBuilder = null;
            String s;

            // run the Unix "python script to collect metrics" command
            // using the Runtime exec method:
            // TODO: Come up with a better way for the location of the script
            ProcessBuilder pb = new ProcessBuilder("python", System.getProperty("user.dir") +"/src/main/resources/get_metric.py", System.getenv("HOME"));
            Process p = pb.start();
            System.out.println(System.getProperty("user.dir"));
            BufferedReader stdOut = new BufferedReader(new
                    InputStreamReader(p.getInputStream()));

            BufferedReader stdError = new BufferedReader(new
                    InputStreamReader(p.getErrorStream()));

            // read the output from the command
            while ((s = stdOut.readLine()) != null) {
                if (this.stringBuilder == null) {
                    this.stringBuilder = new StringBuilder(s);
                } else {
                    this.stringBuilder.append(s);
                }
                System.out.println(s);
            }
            if (this.stringBuilder == null){
                throw new IOException("Did not get anything from file run on stdout");
            }
            this.jsonString = this.stringBuilder.toString();
            // read any errors from the attempted command
            while (stdError.readLine() != null) {
                throw new IOException("Error while running the python script");
            }
        }
    }


