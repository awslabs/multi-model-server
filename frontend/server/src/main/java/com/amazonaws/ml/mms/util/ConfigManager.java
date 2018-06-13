package com.amazonaws.ml.mms.util;

import io.netty.channel.Channel;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollDomainSocketChannel;
import io.netty.channel.epoll.EpollEventLoopGroup;
import io.netty.channel.kqueue.KQueue;
import io.netty.channel.kqueue.KQueueDomainSocketChannel;
import io.netty.channel.kqueue.KQueueEventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.ssl.SslContext;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

public final class ConfigManager {

    private static final String DEBUG = "debug";
    private static final String PORT = "port";
    private static final String MODEL_SERVER_HOME = "model_server_home";
    private static final String MODEL_STORE = "model_store";
    private static final String LOAD_MODELS = "load_models";
    private static final String NUMBER_OF_NETTY_THREADS = "number_of_netty_threads";
    private static final String MAX_WORKERS = "max_workers";
    private static final String JOB_QUEUE_SIZE = "job_queue_size";
    private static final String MAX_BATCH_SIZE = "batch_size";
    private static final String MAX_BATCH_DELAY = "batch_delay";
    private static final String NUMBER_OF_GPU = "number_of_gpu";
    private static final String NETTY_IO_RATIO = "netty_io_ratio";

    private Properties prop;

    public ConfigManager() {
        prop = new Properties();

        String filePath = System.getenv("MMS_CONFIG_FILE");
        if (filePath == null) {
            filePath = System.getProperty("mmsConfigFile", "config.properties");
        }

        File file = new File(filePath);
        if (file.exists()) {
            try (FileInputStream stream = new FileInputStream(file)) {
                prop.load(stream);
            } catch (IOException e) {
                throw new IllegalStateException("Unable to read configuration file", e);
            }
        }
    }

    public boolean isDebug() {
        return Boolean.parseBoolean(prop.getProperty(DEBUG, "false"));
    }

    public int getPort() {
        return getIntProperty(PORT, 8080);
    }

    public int getNettyThreads() {
        return getIntProperty(NUMBER_OF_NETTY_THREADS, 0);
    }

    public int getMaxWorkers() {
        return getIntProperty(MAX_WORKERS, Runtime.getRuntime().availableProcessors());
    }

    public int getJobQueueSize() {
        return getIntProperty(JOB_QUEUE_SIZE, 100);
    }

    public int getMaxBatchSize() {
        return getIntProperty(MAX_BATCH_SIZE, 1);
    }

    public int getMaxBatchDelay() {
        return getIntProperty(MAX_BATCH_DELAY, 100);
    }

    public int getNumberOfGpu() {
        return getIntProperty(NUMBER_OF_GPU, 0);
    }

    public String getModelServerHome() {
        String mmsHome = System.getenv("MODEL_SERVER_HOME");
        if (mmsHome == null) {
            mmsHome = System.getProperty(MODEL_SERVER_HOME);
            if (mmsHome == null) {
                File dir = new File("/mxnet-model-server");
                if (!dir.exists()) {
                    dir = new File(".");
                }
                mmsHome = getProperty(MODEL_SERVER_HOME, dir.getAbsolutePath());
            }
        }
        return mmsHome;
    }

    public String getModelStore() {
        String mmsHome = getModelServerHome();
        return getProperty(MODEL_STORE, mmsHome + "/model");
    }

    public String getLoadModels() {
        return getProperty(LOAD_MODELS, "ALL");
    }

    public SslContext getSslContext() {
        return null;
    }

    public EventLoopGroup newEventLoopGroup(int threads, boolean adjustIoRatio) {
        if (Epoll.isAvailable()) {
            return new EpollEventLoopGroup(threads);
        } else if (KQueue.isAvailable()) {
            return new KQueueEventLoopGroup(threads);
        }

        NioEventLoopGroup group = new NioEventLoopGroup(threads);
        if (adjustIoRatio) {
            group.setIoRatio(getIntProperty(NETTY_IO_RATIO, 50));
        }
        return group;
    }

    public Class<? extends Channel> getChannel() {
        if (Epoll.isAvailable()) {
            return EpollDomainSocketChannel.class;
        } else if (KQueue.isAvailable()) {
            return KQueueDomainSocketChannel.class;
        }

        return NioSocketChannel.class;
    }

    public String getProperty(String key, String def) {
        return prop.getProperty(key, def);
    }

    private int getIntProperty(String key, int def) {
        String value = prop.getProperty(key);
        if (value == null) {
            return def;
        }
        return Integer.parseInt(value);
    }
}
