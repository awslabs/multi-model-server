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
import com.amazonaws.ml.mms.wlm.WorkerInitializationException;
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
import java.io.FileFilter;
import java.io.IOException;
import java.security.GeneralSecurityException;
import java.util.Collection;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.FileFilterUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelServer {

    private Logger logger = LoggerFactory.getLogger(ModelServer.class);

    private ServerGroups serverGroups;
    private ChannelFuture future;
    private AtomicBoolean stopped = new AtomicBoolean(false);

    private ConfigManager configManager;

    /** Creates a new {@code ModelServer} instance. */
    public ModelServer(ConfigManager configManager) {
        this.configManager = configManager;
        serverGroups = new ServerGroups(configManager);
    }

    public static void main(String[] args)
            throws InterruptedException, InvalidModelException, WorkerInitializationException,
                    IOException, GeneralSecurityException {
        ConfigManager configManager = new ConfigManager();

        InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);
        new ModelServer(configManager).startAndWait();
    }

    @SuppressWarnings("PMD.SystemPrintln")
    public void startAndWait()
            throws InterruptedException, InvalidModelException, WorkerInitializationException,
                    IOException, GeneralSecurityException {
        try {
            initModelStore();

            ChannelFuture f = start();
            // Create and schedule metrics manager
            MetricManager.scheduleMetrics(configManager);
            System.out.println("Model server started.");
            f.sync();
        } finally {
            serverGroups.shutdown(true);
            logger.info("Model server stopped.");
        }
        Runtime.getRuntime().halt(-1); // NOPMD
    }

    public void initModelStore() throws InvalidModelException, WorkerInitializationException {
        logger.debug("Loading initial models...");
        WorkLoadManager wlm = new WorkLoadManager(configManager, serverGroups.getBackendGroup());
        ModelManager.init(configManager, wlm);

        File modelStore = new File(configManager.getModelStore());
        if (modelStore.exists()) {
            ModelManager modelManager = ModelManager.getInstance();
            String loadModels = configManager.getLoadModels();
            if ("ALL".equalsIgnoreCase(loadModels)) {
                String[] extensions = new String[] {"model", "mar"};
                Collection<File> models = FileUtils.listFiles(modelStore, extensions, false);
                for (File modelFile : models) {
                    ModelArchive archive = modelManager.registerModel(modelFile.getName());
                    modelManager.updateModel(archive.getModelName(), 1, 1);
                }
                // Check folders to see if they can be models as well
                File[] dirs =
                        modelStore.listFiles((FileFilter) FileFilterUtils.directoryFileFilter());
                if (dirs != null) {
                    for (File dir : dirs) {
                        ModelArchive archive = modelManager.registerModel(dir.getName());
                        modelManager.updateModel(archive.getModelName(), 1, 1);
                    }
                }
            } else {
                String[] models = loadModels.split(",");
                for (String model : models) {
                    File modelFile = new File(modelStore, model);
                    if (!modelFile.exists()) {
                        if (!model.endsWith(".model")) {
                            modelFile = new File(modelStore, model + ".model");
                        }

                        if (!model.endsWith(".mar")) {
                            modelFile = new File(modelStore, model + ".mar");
                        }
                    }

                    if (modelFile.exists()) {
                        ModelArchive archive = modelManager.registerModel(modelFile.getName());
                        modelManager.updateModel(archive.getModelName(), 1, 1);
                    }
                }
            }
        }
    }

    /**
     * Main Method that prepares the future for the channel and sets up the ServerBootstrap.
     *
     * @return A ChannelFuture object
     * @throws InterruptedException if interrupted
     */
    public ChannelFuture start()
            throws InterruptedException, IOException, GeneralSecurityException {
        stopped.set(false);

        SslContext sslCtx = configManager.getSslContext();
        int port = configManager.getPort();

        EventLoopGroup serverGroup = serverGroups.getServerGroup();
        EventLoopGroup workerGroup = serverGroups.getChildGroup();

        Class<? extends ServerChannel> channelClass = NettyUtils.getServerChannel();

        ServerBootstrap b = new ServerBootstrap();
        b.option(ChannelOption.SO_BACKLOG, 1024)
                .channel(channelClass)
                .childOption(ChannelOption.SO_LINGER, 0)
                .childOption(ChannelOption.SO_REUSEADDR, true)
                .childOption(ChannelOption.SO_KEEPALIVE, true);
        b.group(serverGroup, workerGroup);
        b.childHandler(new ServerInitializer(sslCtx));
        future = b.bind(port).sync();
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

        logger.info("Initialize server with: {}.", channelClass.getSimpleName());

        future.sync();

        ChannelFuture f = future.channel().closeFuture();
        f.addListener((ChannelFutureListener) future -> logger.info("Model server stopped."));

        logger.info("Listening on port: {}", port);
        return f;
    }

    public boolean isRunning() {
        return !stopped.get();
    }

    public void stop() {
        if (stopped.get()) {
            return;
        }

        stopped.set(true);
        future.channel().close();
        serverGroups.shutdown(true);
        serverGroups.init();
    }
}
