package com.sagemaker.endpoint;

import com.amazonaws.ml.mms.servingsdk.Context;
import com.amazonaws.ml.mms.servingsdk.ModelServerEndpoint;
import com.amazonaws.ml.mms.servingsdk.annotations.Endpoint;
import com.amazonaws.ml.mms.servingsdk.annotations.helpers.EndpointTypes;
import com.amazonaws.ml.mms.servingsdk.http.Request;
import com.amazonaws.ml.mms.servingsdk.http.Response;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.Properties;

@Endpoint(
        urlPattern = "execution-parameters",
        endpointType = EndpointTypes.INFERENCE,
        description = "Execution parameters endpoint")
public class ExecutionParameters extends ModelServerEndpoint {
    @Override
    public void doGet(Request req, Response rsp, Context ctx) throws IOException {
        Properties prop = ctx.getConfig();
        HashMap<String, String> r = new HashMap<>();
        r.put("MAX_CONCURRENT_TRANSFORMS", prop.getProperty("NUM_WORKERS", "1"));
        r.put("BATCH_STRATEGY", "SINGLE_RECORD");
        r.put("MAX_PAYLOAD_IN_MB", prop.getProperty("max_request_size"));
        r.put("BATCH", "true");
        rsp.getOutputStream()
                .write(
                        new GsonBuilder()
                                .setPrettyPrinting()
                                .create()
                                .toJson(r)
                                .getBytes(Charset.forName("UTF-8")));
    }
}
