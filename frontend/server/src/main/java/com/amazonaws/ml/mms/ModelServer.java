/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.amazonaws.ml.mms;

import com.amazonaws.ml.mms.archive.InvalidModelException;
import com.amazonaws.ml.mms.archive.ModelArchive;
import com.amazonaws.ml.mms.metrics.MetricManager;
import com.amazonaws.ml.mms.util.ConfigManager;
import com.amazonaws.ml.mms.util.NettyUtils;
import com.amazonaws.ml.mms.util.ServerGroups;
import com.amazonaws.ml.mms.wlm.ModelManager;
import com.amazonaws.ml.mms.wlm.WorkLoadManager;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.ServerChannel;
import io.netty.handler.ssl.SslContext;
import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.security.GeneralSecurityException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelServer {

    private Logger logger = LoggerFactory.getLogger(ModelServer.class);

    private ServerGroups serverGroups;
    private List<ChannelFuture> futures;
    private AtomicBoolean stopped = new AtomicBoolean(false);

    private ConfigManager configManager;

    /** Creates a new {@code ModelServer} instance. */
    public ModelServer(ConfigManager configManager) {
        this.configManager = configManager;
        serverGroups = new ServerGroups(configManager);
    }

    public static void main(String[] args)
            throws InterruptedException, IOException, GeneralSecurityException {
        Options options = ConfigManager.Arguments.getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            ConfigManager.Arguments arguments = new ConfigManager.Arguments(cmd);

            ConfigManager configManager = new ConfigManager(arguments);

            InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);
            new ModelServer(configManager).startAndWait();
        } catch (IllegalArgumentException e) {
            System.out.println("Invalid configuration: " + e.getMessage()); // NOPMD
        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setLeftPadding(1);
            formatter.setWidth(120);
            formatter.printHelp(e.getMessage(), options);
            System.exit(1); // NOPMD
        }
    }

    public void startAndWait() throws InterruptedException, IOException, GeneralSecurityException {
        try {
            List<ChannelFuture> channelFutures = start();
            // Create and schedule metrics manager
            MetricManager.scheduleMetrics(configManager);
            System.out.println("Model server started."); // NOPMD
            channelFutures.get(0).sync();
        } finally {
            serverGroups.shutdown(true);
            logger.info("Model server stopped.");
        }
        Runtime.getRuntime().halt(-1); // NOPMD
    }

    private void initModelStore() throws IOException {
        WorkLoadManager wlm = new WorkLoadManager(configManager, serverGroups.getBackendGroup());
        ModelManager.init(configManager, wlm);

        String loadModels = configManager.getLoadModels();
        if (loadModels == null || loadModels.isEmpty()) {
            return;
        }

        ModelManager modelManager = ModelManager.getInstance();
        if ("ALL".equalsIgnoreCase(loadModels)) {
            String modelStore = configManager.getModelStore();
            if (modelStore == null) {
                logger.warn("Model store is not configured.");
                return;
            }

            File modelStoreDir = new File(modelStore);
            if (!modelStoreDir.exists()) {
                logger.warn("Model store path is not found: {}", modelStore);
                return;
            }

            // Check folders to see if they can be models as well
            File[] files = modelStoreDir.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isHidden()) {
                        continue;
                    }
                    String fileName = file.getName();
                    if (file.isFile()
                            && !fileName.endsWith(".mar")
                            && !fileName.endsWith(".model")) {
                        continue;
                    }
                    try {
                        logger.debug("Loading models from model store: {}", file.getName());

                        ModelArchive archive = modelManager.registerModel(file.getName());
                        modelManager.updateModel(archive.getModelName(), 1, 1);
                    } catch (InvalidModelException e) {
                        logger.warn("Failed to load model: " + file.getAbsolutePath(), e);
                    }
                }
            }
            return;
        }

        String[] models = loadModels.split(",");
        for (String model : models) {
            String[] pair = model.split("=", 2);
            String modelName = null;
            String url;
            if (pair.length == 1) {
                url = pair[0];
            } else {
                modelName = pair[0];
                url = pair[1];
            }
            if (url.isEmpty()) {
                continue;
            }

            try {
                logger.debug("Loading initial models: {}", url);

                ModelArchive archive =
                        modelManager.registerModel(url, modelName, null, null, 1, 100);
                modelManager.updateModel(archive.getModelName(), 1, 1);
            } catch (InvalidModelException e) {
                logger.warn("Failed to load model: " + url, e);
            }
        }
    }

    public ChannelFuture initializeServer(
            URI address,
            boolean management,
            EventLoopGroup serverGroup,
            EventLoopGroup workerGroup,
            Class<? extends ServerChannel> channelClass)
            throws InterruptedException, IOException, GeneralSecurityException {
        final String purpose = management ? "Management" : "Inference";
        ServerBootstrap b = new ServerBootstrap();
        b.option(ChannelOption.SO_BACKLOG, 1024)
                .channel(channelClass)
                .childOption(ChannelOption.SO_LINGER, 0)
                .childOption(ChannelOption.SO_REUSEADDR, true)
                .childOption(ChannelOption.SO_KEEPALIVE, true);
        b.group(serverGroup, workerGroup);

        SslContext sslCtx = null;
        if ("https".equalsIgnoreCase(address.getScheme())) {
            sslCtx = configManager.getSslContext();
        }
        b.childHandler(new ServerInitializer(sslCtx, management));

        ChannelFuture future = b.bind(address.getHost(), address.getPort()).sync();
        future.addListener(
                (ChannelFutureListener)
                        f -> {
                            if (!f.isSuccess()) {
                                try {
                                    f.get();
                                } catch (InterruptedException | ExecutionException e) {
                                    logger.error("", e);
                                }
                                System.exit(-1); // NO PMD
                            }
                            serverGroups.registerChannel(f.channel());
                        });

        future.sync();

        ChannelFuture f = future.channel().closeFuture();
        f.addListener(
                (ChannelFutureListener)
                        listener -> logger.info("{} model server stopped.", purpose));

        logger.info("{} API listening on port: {}", purpose, address.getPort());
        return f;
    }

    /**
     * Main Method that prepares the future for the channel and sets up the ServerBootstrap.
     *
     * @return A ChannelFuture object
     * @throws InterruptedException if interrupted
     */
    public List<ChannelFuture> start()
            throws InterruptedException, IOException, GeneralSecurityException {
        stopped.set(false);

        logger.info(configManager.dumpConfigurations());

        initModelStore();

        URI inferenceAddress = configManager.getInferenceAddress();
        URI managementAddress = configManager.getManagementAddress();
        if (inferenceAddress.getPort() == managementAddress.getPort()) {
            throw new IllegalArgumentException(
                    "Inference port must differ from the management port");
        }

        EventLoopGroup serverGroup = serverGroups.getServerGroup();
        EventLoopGroup workerGroup = serverGroups.getChildGroup();

        Class<? extends ServerChannel> channelClass = NettyUtils.getServerChannel();
        logger.info("Initialize servers with: {}.", channelClass.getSimpleName());

        futures =
                Arrays.asList(
                        initializeServer(
                                inferenceAddress, false, serverGroup, workerGroup, channelClass),
                        initializeServer(
                                managementAddress, true, serverGroup, workerGroup, channelClass));

        return futures;
    }

    public boolean isRunning() {
        return !stopped.get();
    }

    public void stop() {
        if (stopped.get()) {
            return;
        }

        stopped.set(true);
        for (ChannelFuture future : futures) {
            future.channel().close();
        }
        serverGroups.shutdown(true);
        serverGroups.init();
    }
}
