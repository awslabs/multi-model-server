package software.amazon.ai.mms.plugins.endpoint;

import com.google.gson.GsonBuilder;
import com.google.gson.annotations.SerializedName;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Properties;
import software.amazon.ai.mms.servingsdk.Context;
import software.amazon.ai.mms.servingsdk.ModelServerEndpoint;
import software.amazon.ai.mms.servingsdk.annotations.Endpoint;
import software.amazon.ai.mms.servingsdk.annotations.helpers.EndpointTypes;
import software.amazon.ai.mms.servingsdk.http.Request;
import software.amazon.ai.mms.servingsdk.http.Response;

@Endpoint(
        urlPattern = "execution-parameters",
        endpointType = EndpointTypes.INFERENCE,
        description = "Execution parameters endpoint")
public class ExecutionParameters extends ModelServerEndpoint {

    @Override
    public void doGet(Request req, Response rsp, Context ctx) throws IOException {
        Properties prop = ctx.getConfig();
        // 6 * 1024 * 1024
        int maxRequestSize = Integer.parseInt(prop.getProperty("max_request_size", "6291456"));
        ExecutionParametersResponse r = new ExecutionParametersResponse();
        r.setMaxConcurrentTransforms(Integer.parseInt(prop.getProperty("NUM_WORKERS", "1")));
        r.setBatchStrategy("MULTI_RECORD");
        r.setMaxPayloadInMB(maxRequestSize / (1024 * 1024));
        rsp.getOutputStream()
                .write(
                        new GsonBuilder()
                                .setPrettyPrinting()
                                .create()
                                .toJson(r)
                                .getBytes(StandardCharsets.UTF_8));
    }

    /** Response for Model server endpoint */
    public static class ExecutionParametersResponse {
        @SerializedName("MaxConcurrentTransforms")
        private int maxConcurrentTransforms;

        @SerializedName("BatchStrategy")
        private String batchStrategy;

        @SerializedName("MaxPayloadInMB")
        private int maxPayloadInMB;

        public ExecutionParametersResponse() {
            maxConcurrentTransforms = 4;
            batchStrategy = "MULTI_RECORD";
            maxPayloadInMB = 6;
        }

        public int getMaxConcurrentTransforms() {
            return maxConcurrentTransforms;
        }

        public String getBatchStrategy() {
            return batchStrategy;
        }

        public int getMaxPayloadInMB() {
            return maxPayloadInMB;
        }

        public void setMaxConcurrentTransforms(int newMaxConcurrentTransforms) {
            maxConcurrentTransforms = newMaxConcurrentTransforms;
        }

        public void setBatchStrategy(String newBatchStrategy) {
            batchStrategy = newBatchStrategy;
        }

        public void setMaxPayloadInMB(int newMaxPayloadInMB) {
            maxPayloadInMB = newMaxPayloadInMB;
        }
    }
}
