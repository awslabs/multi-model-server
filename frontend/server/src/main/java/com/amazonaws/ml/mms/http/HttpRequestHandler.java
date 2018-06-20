package com.amazonaws.ml.mms.http;

import com.amazonaws.ml.mms.archive.InvalidModelException;
import com.amazonaws.ml.mms.openapi.OpenApiUtils;
import com.amazonaws.ml.mms.util.JsonUtils;
import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.wlm.Job;
import com.amazonaws.ml.mms.wlm.Model;
import com.amazonaws.ml.mms.wlm.ModelManager;
import com.amazonaws.ml.mms.wlm.Payload;
import com.amazonaws.ml.mms.wlm.WorkerInitializationException;
import com.google.gson.JsonParseException;
import io.netty.buffer.ByteBufInputStream;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class handling inbound HTTP requests.
 *
 * <p>This class
 */
public class HttpRequestHandler extends SimpleChannelInboundHandler<FullHttpRequest> {

    private static final Logger logger = LoggerFactory.getLogger(HttpRequestHandler.class);

    /** Creates a new {@code HttpRequestHandler} instance. */
    public HttpRequestHandler() {}

    /** {@inheritDoc} */
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest req) {
        if (!req.decoderResult().isSuccess()) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        try {
            handleRequest(ctx, req);
        } catch (Throwable t) {
            logger.error("", t);
            NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        logger.error("", cause);
        ctx.close();
    }

    private void handleRequest(ChannelHandlerContext ctx, FullHttpRequest req) {
        QueryStringDecoder decoder = new QueryStringDecoder(req.uri());
        String body;
        switch (decoder.path()) {
            case "/":
                body = "OK";
                break;
            case "/ping":
                body = "{\"status\":\"healthy\"}";
                break;
            case "/api-description":
                handleOpenApi(ctx, req);
                return;
            case "/invocations":
                handleInvocations(ctx, decoder, req);
                return;
            case "/register":
                handleRegisterModel(ctx, decoder, req);
                return;
            case "/unregister":
                handleUnregisterModel(ctx, decoder, req);
                return;
            case "/scale":
                handleScaleModel(ctx, decoder, req);
                return;
            default:
                NettyUtils.sendError(ctx, HttpResponseStatus.NOT_FOUND);
                return;
        }

        NettyUtils.sendJsonResponse(ctx, body);
    }

    private void handleOpenApi(ChannelHandlerContext ctx, FullHttpRequest req) {
        String scheme = ctx.pipeline().get("ssl") == null ? "http" : "https";
        String host = req.headers().get(HttpHeaderNames.HOST);
        ModelManager modelManager = ModelManager.getInstance();
        Map<String, Model> models = modelManager.getModels();
        String resp = OpenApiUtils.getMetadata(host, scheme, models);
        NettyUtils.sendJsonResponse(ctx, resp);
    }

    private void handleInvocations(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, FullHttpRequest req) {
        String modelName = NettyUtils.getParameter(decoder, "model_name", null);
        String data = NettyUtils.getParameter(decoder, "data", null);

        HttpMethod method = req.method();
        if (method == HttpMethod.GET) {
            if (data == null) {
                NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
                return;
            }
            Payload payload = new Payload(modelName, data);
            Job job = new Job(ctx, payload);
            logger.debug("received request: {}", job.getJobId());
            HttpResponseStatus status = ModelManager.getInstance().addJob(job);
            if (status != HttpResponseStatus.OK) {
                NettyUtils.sendError(ctx, status);
            }
            return;
        }

        if (method != HttpMethod.POST) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        Payload payload = new Payload(modelName, data);
        CharSequence contentType = HttpUtil.getMimeType(req);
        if (HttpHeaderValues.APPLICATION_JSON.contentEqualsIgnoreCase(contentType)) {
            try (Reader reader =
                    new InputStreamReader(
                            new ByteBufInputStream(req.content()), StandardCharsets.UTF_8)) {
                Param p = JsonUtils.GSON.fromJson(reader, Param.class);
                if (p.modelName != null) {
                    payload.setId(p.modelName);
                }
                if (p.data != null) {
                    payload.setData(p.data);
                }
            } catch (IOException | JsonParseException e) {
                NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
                return;
            }
        } else if (HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED.contentEqualsIgnoreCase(
                        contentType)
                || HttpPostRequestDecoder.isMultipart(req)) {
            HttpPostRequestDecoder form = new HttpPostRequestDecoder(req);
            try {
                modelName = NettyUtils.getFormField(form.getBodyHttpData("model_name"));
                byte[] buf = NettyUtils.getFormData(form.getBodyHttpData("data"));
                if (modelName != null) {
                    payload.setId(modelName);
                }
                if (buf != null) {
                    payload.setData(buf);
                }
            } catch (IllegalArgumentException e) {
                NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
                return;
            } finally {
                form.cleanFiles();
            }
        } else {
            byte[] buf = NettyUtils.getBytes(req.content());
            payload.setData(buf);
        }

        Job job = new Job(ctx, payload);
        HttpResponseStatus status = ModelManager.getInstance().addJob(job);
        if (status != HttpResponseStatus.OK) {
            NettyUtils.sendError(ctx, status);
        }
    }

    private void handleRegisterModel(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, FullHttpRequest req) {
        HttpMethod method = req.method();
        if (method != HttpMethod.GET) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        String modelUrl = NettyUtils.getParameter(decoder, "url", null);

        if (modelUrl == null) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        ModelManager modelManager = ModelManager.getInstance();
        try {
            modelManager.registerModel(modelUrl);
        } catch (InvalidModelException e) {
            logger.warn("Failed to load model", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        NettyUtils.sendJsonResponse(ctx, "{\"status\":\"Model registered\"}");
    }

    private void handleUnregisterModel(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, FullHttpRequest req) {
        HttpMethod method = req.method();
        if (method != HttpMethod.GET) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        String modelName = NettyUtils.getParameter(decoder, "model_name", null);
        if (modelName == null) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        ModelManager modelManager = ModelManager.getInstance();
        try {
            if (!modelManager.unregisterModel(modelName)) {
                NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            }
        } catch (WorkerInitializationException e) {
            logger.warn("Failed to load model", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        NettyUtils.sendJsonResponse(ctx, "{\"status\":\"Model unregistered\"}");
    }

    private void handleScaleModel(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, FullHttpRequest req) {
        String modelName = NettyUtils.getParameter(decoder, "model_name", null);
        int minWorkers = NettyUtils.getIntParameter(decoder, "min_worker", 1);
        int maxWorkers = NettyUtils.getIntParameter(decoder, "max_worker", 1);

        HttpMethod method = req.method();
        if (method != HttpMethod.GET) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        if (modelName == null) {
            NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
            return;
        }

        ModelManager modelManager = ModelManager.getInstance();
        try {
            if (!modelManager.updateModel(modelName, minWorkers, maxWorkers)) {
                NettyUtils.sendError(ctx, HttpResponseStatus.BAD_REQUEST);
                return;
            }
        } catch (WorkerInitializationException e) {
            logger.error("Failed update model.", e);
            NettyUtils.sendError(ctx, HttpResponseStatus.INTERNAL_SERVER_ERROR);
            return;
        }

        NettyUtils.sendJsonResponse(ctx, "{\"status\":\"Worker updated\"}");
    }

    public static final class Param {

        String modelName;
        String data;

        public Param() {}

        public Param(String modelName, String data) {
            this.modelName = modelName;
            this.data = data;
        }
    }
}
