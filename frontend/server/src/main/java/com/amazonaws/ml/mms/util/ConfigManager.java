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
package com.amazonaws.ml.mms.util;

import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.SelfSignedCertificate;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.KeyException;
import java.security.KeyFactory;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collection;
import java.util.Enumeration;
import java.util.InvalidPropertiesFormatException;
import java.util.List;
import java.util.Properties;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.io.IOUtils;
import org.apache.log4j.Appender;
import org.apache.log4j.AsyncAppender;
import org.apache.log4j.Logger;

public final class ConfigManager {

    public static final String MODEL_METRICS_LOGGER = "MODEL_METRICS";
    public static final String MODEL_LOGGER = "MODEL_LOG";
    public static final String MMS_METRICS_LOGGER = "MMS_METRICS";

    private static final String DEBUG = "debug";
    private static final String INFERENCE_ADDRESS = "inference_address";
    private static final String MANAGEMENT_ADDRESS = "management_address";
    private static final String MODEL_SERVER_HOME = "model_server_home";
    private static final String MODEL_STORE = "model_store";
    private static final String LOAD_MODELS = "load_models";
    private static final String BLACKLIST_ENV_VARS = "blacklist_env_vars";
    private static final String DEFAULT_WORKERS_PER_MODEL = "default_workers_per_model";

    // advanced parameters for performance tuning
    private static final String NUMBER_OF_NETTY_THREADS = "number_of_netty_threads";
    private static final String NETTY_CLIENT_THREADS = "netty_client_threads";
    private static final String USE_NATIVE_IO = "use_native_io";
    private static final String IO_RATIO = "io_ratio";
    private static final String JOB_QUEUE_SIZE = "job_queue_size";

    private static final String NUMBER_OF_GPU = "number_of_gpu";
    private static final String METRIC_TIME_INTERVAL = "metric_time_interval";
    private static final String ASYNC_LOGGING = "async_logging";

    private static final String CORS_ALLOWED_ORIGIN = "cors_allowed_origin";
    private static final String CORS_ALLOWED_METHODS = "cors_allowed_methods";
    private static final String CORS_ALLOWED_HEADERS = "cors_allowed_headers";

    private static final String KEYSTORE = "keystore";
    private static final String KEYSTORE_PASS = "keystore_pass";
    private static final String KEYSTORE_TYPE = "keystore_type";
    private static final String CERTIFICATE_FILE = "certificate_file";
    private static final String PRIVATE_KEY_FILE = "private_key_file";

    private Pattern blacklistPattern;
    private Properties prop;

    private static ConfigManager instance;

    private String hostName;

    private ConfigManager(Arguments args) {
        prop = new Properties();

        String filePath = System.getenv("MMS_CONFIG_FILE");
        if (filePath == null) {
            filePath = args.getMmsConfigFile();
            if (filePath == null) {
                filePath = System.getProperty("mmsConfigFile", "config.properties");
            }
        }

        File file = new File(filePath);
        if (file.exists()) {
            try (FileInputStream stream = new FileInputStream(file)) {
                prop.load(stream);
                prop.put("mmsConfigFile", filePath);
            } catch (IOException e) {
                throw new IllegalStateException("Unable to read configuration file", e);
            }
        }
        String logLocation = System.getenv("LOG_LOCATION");
        if (logLocation != null) {
            System.setProperty("LOG_LOCATION", logLocation);
        } else if (System.getProperty("LOG_LOCATION") == null) {
            System.setProperty("LOG_LOCATION", "logs");
        }

        String metricsLocation = System.getenv("METRICS_LOCATION");
        if (metricsLocation != null) {
            System.setProperty("METRICS_LOCATION", metricsLocation);
        } else if (System.getProperty("METRICS_LOCATION") == null) {
            System.setProperty("METRICS_LOCATION", "logs");
        }

        String modelStore = args.getModelStore();
        if (modelStore != null) {
            prop.setProperty(MODEL_STORE, modelStore);
        }

        String[] models = args.getModels();
        if (models != null) {
            prop.setProperty(LOAD_MODELS, String.join(",", models));
        }

        prop.setProperty(
                NUMBER_OF_GPU,
                String.valueOf(
                        Integer.min(
                                getAvailableGpu(),
                                getIntProperty(NUMBER_OF_GPU, Integer.MAX_VALUE))));

        String pythonExecutable = args.getPythonExecutable();
        if (pythonExecutable != null) {
            prop.setProperty("PYTHON_EXECUTABLE", pythonExecutable);
        }

        try {
            InetAddress ip = InetAddress.getLocalHost();
            hostName = ip.getHostName();
        } catch (UnknownHostException e) {
            hostName = "Unknown";
        }

        if (Boolean.parseBoolean(prop.getProperty(ASYNC_LOGGING))) {
            enableAsyncLogging();
        }
    }

    public String getHostName() {
        return hostName;
    }

    public static void init(Arguments args) {
        instance = new ConfigManager(args);
    }

    public static ConfigManager getInstance() {
        return instance;
    }

    public boolean isDebug() {
        return Boolean.getBoolean("DEBUG")
                || Boolean.parseBoolean(prop.getProperty(DEBUG, "false"));
    }

    public Connector getListener(boolean management) {
        String binding;
        if (management) {
            binding = prop.getProperty(MANAGEMENT_ADDRESS, "http://127.0.0.1:8081");
        } else {
            binding = prop.getProperty(INFERENCE_ADDRESS, "http://127.0.0.1:8080");
        }
        return Connector.parse(binding, management);
    }

    public int getNettyThreads() {
        return getIntProperty(NUMBER_OF_NETTY_THREADS, 0);
    }

    public int getNettyClientThreads() {
        return getIntProperty(NETTY_CLIENT_THREADS, 0);
    }

    public int getJobQueueSize() {
        return getIntProperty(JOB_QUEUE_SIZE, 100);
    }

    public int getNumberOfGpu() {
        return getIntProperty(NUMBER_OF_GPU, 0);
    }

    public int getDefaultWorkers() {
        if (isDebug()) {
            return 1;
        }

        int workers = getIntProperty(DEFAULT_WORKERS_PER_MODEL, 0);
        if (workers == 0) {
            workers = getNumberOfGpu();
        }
        if (workers == 0) {
            workers = Runtime.getRuntime().availableProcessors();
        }
        return workers;
    }

    public int getMetricTimeInterval() {
        return getIntProperty(METRIC_TIME_INTERVAL, 60);
    }

    public String getModelServerHome() {
        String mmsHome = System.getenv("MODEL_SERVER_HOME");
        if (mmsHome == null) {
            mmsHome = System.getProperty(MODEL_SERVER_HOME);
            if (mmsHome == null) {
                mmsHome = getProperty(MODEL_SERVER_HOME, null);
                if (mmsHome == null) {
                    mmsHome = getCanonicalPath(findMmsHome());
                    return mmsHome;
                }
            }
        }

        File dir = new File(mmsHome);
        if (!dir.exists()) {
            throw new IllegalArgumentException("Model server home not exist: " + mmsHome);
        }
        mmsHome = getCanonicalPath(dir);
        return mmsHome;
    }

    public String getPythonExecutable() {
        return prop.getProperty("PYTHON_EXECUTABLE", "python");
    }

    public String getModelStore() {
        return getCanonicalPath(prop.getProperty(MODEL_STORE));
    }

    public String getLoadModels() {
        return prop.getProperty(LOAD_MODELS);
    }

    public Pattern getBlacklistPattern() {
        return blacklistPattern;
    }

    public String getCorsAllowedOrigin() {
        return prop.getProperty(CORS_ALLOWED_ORIGIN);
    }

    public String getCorsAllowedMethods() {
        return prop.getProperty(CORS_ALLOWED_METHODS);
    }

    public String getCorsAllowedHeaders() {
        return prop.getProperty(CORS_ALLOWED_HEADERS);
    }

    public SslContext getSslContext() throws IOException, GeneralSecurityException {
        List<String> supportedCiphers =
                Arrays.asList(
                        "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
                        "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256");

        PrivateKey privateKey;
        X509Certificate[] chain;
        String keyStoreFile = prop.getProperty(KEYSTORE);
        String privateKeyFile = prop.getProperty(PRIVATE_KEY_FILE);
        String certificateFile = prop.getProperty(CERTIFICATE_FILE);
        if (keyStoreFile != null) {
            char[] keystorePass = getProperty(KEYSTORE_PASS, "changeit").toCharArray();
            String keystoreType = getProperty(KEYSTORE_TYPE, "PKCS12");
            KeyStore keyStore = KeyStore.getInstance(keystoreType);
            try (InputStream is = new FileInputStream(keyStoreFile)) {
                keyStore.load(is, keystorePass);
            }

            Enumeration<String> en = keyStore.aliases();
            String keyAlias = null;
            while (en.hasMoreElements()) {
                String alias = en.nextElement();
                if (keyStore.isKeyEntry(alias)) {
                    keyAlias = alias;
                    break;
                }
            }

            if (keyAlias == null) {
                throw new KeyException("No key entry found in keystore.");
            }

            privateKey = (PrivateKey) keyStore.getKey(keyAlias, keystorePass);

            Certificate[] certs = keyStore.getCertificateChain(keyAlias);
            chain = new X509Certificate[certs.length];
            for (int i = 0; i < certs.length; ++i) {
                chain[i] = (X509Certificate) certs[i];
            }
        } else if (privateKeyFile != null && certificateFile != null) {
            privateKey = loadPrivateKey(privateKeyFile);
            chain = loadCertificateChain(certificateFile);
        } else {
            SelfSignedCertificate ssc = new SelfSignedCertificate();
            privateKey = ssc.key();
            chain = new X509Certificate[] {ssc.cert()};
        }

        return SslContextBuilder.forServer(privateKey, chain)
                .protocols("TLSv1.2")
                .ciphers(supportedCiphers)
                .build();
    }

    private PrivateKey loadPrivateKey(String keyFile) throws IOException, GeneralSecurityException {
        KeyFactory keyFactory = KeyFactory.getInstance("RSA");
        try (InputStream is = new FileInputStream(keyFile)) {
            String content = IOUtils.toString(is, StandardCharsets.UTF_8);
            content = content.replaceAll("-----(BEGIN|END)( RSA)? PRIVATE KEY-----\\s*", "");
            byte[] buf = Base64.getMimeDecoder().decode(content);
            try {
                PKCS8EncodedKeySpec privKeySpec = new PKCS8EncodedKeySpec(buf);
                return keyFactory.generatePrivate(privKeySpec);
            } catch (InvalidKeySpecException e) {
                // old private key is OpenSSL format private key
                buf = OpenSslKey.convertPrivateKey(buf);
                PKCS8EncodedKeySpec privKeySpec = new PKCS8EncodedKeySpec(buf);
                return keyFactory.generatePrivate(privKeySpec);
            }
        }
    }

    private X509Certificate[] loadCertificateChain(String keyFile)
            throws IOException, GeneralSecurityException {
        CertificateFactory cf = CertificateFactory.getInstance("X.509");
        try (InputStream is = new FileInputStream(keyFile)) {
            Collection<? extends Certificate> certs = cf.generateCertificates(is);
            int i = 0;
            X509Certificate[] chain = new X509Certificate[certs.size()];
            for (Certificate cert : certs) {
                chain[i++] = (X509Certificate) cert;
            }
            return chain;
        }
    }

    public String getProperty(String key, String def) {
        return prop.getProperty(key, def);
    }

    public void validateConfigurations() throws InvalidPropertiesFormatException {
        String blacklistVars = prop.getProperty(BLACKLIST_ENV_VARS, "");
        try {
            blacklistPattern = Pattern.compile(blacklistVars);
        } catch (PatternSyntaxException e) {
            throw new InvalidPropertiesFormatException(e);
        }
    }

    public String dumpConfigurations() {
        Runtime runtime = Runtime.getRuntime();
        return "\nMMS Home: "
                + getModelServerHome()
                + "\nCurrent directory: "
                + getCanonicalPath(".")
                + "\nTemp directory: "
                + System.getProperty("java.io.tmpdir")
                + "\nNumber of GPUs: "
                + getNumberOfGpu()
                + "\nNumber of CPUs: "
                + runtime.availableProcessors()
                + "\nMax heap size: "
                + (runtime.maxMemory() / 1024 / 1024)
                + " M\nPython executable: "
                + (getPythonExecutable() == null ? "N/A" : getPythonExecutable())
                + "\nConfig file: "
                + prop.getProperty("mmsConfigFile", "N/A")
                + "\nInference address: "
                + getListener(false)
                + "\nManagement address: "
                + getListener(true)
                + "\nModel Store: "
                + (getModelStore() == null ? "N/A" : getModelStore())
                + "\nInitial Models: "
                + (getLoadModels() == null ? "N/A" : getLoadModels())
                + "\nLog dir: "
                + getCanonicalPath(System.getProperty("LOG_LOCATION"))
                + "\nMetrics dir: "
                + getCanonicalPath(System.getProperty("METRICS_LOCATION"))
                + "\nNetty threads: "
                + getNettyThreads()
                + "\nNetty client threads: "
                + getNettyClientThreads()
                + "\nDefault workers per model: "
                + getDefaultWorkers()
                + "\nBlacklist Regex: "
                + prop.getProperty(BLACKLIST_ENV_VARS, "N/A");
    }

    public boolean useNativeIo() {
        return Boolean.parseBoolean(prop.getProperty(USE_NATIVE_IO, "true"));
    }

    public int getIoRatio() {
        return getIntProperty(IO_RATIO, 50);
    }

    void setProperty(String key, String value) {
        prop.setProperty(key, value);
    }

    private int getIntProperty(String key, int def) {
        String value = prop.getProperty(key);
        if (value == null) {
            return def;
        }
        return Integer.parseInt(value);
    }

    private File findMmsHome() {
        File cwd = new File(getCanonicalPath("."));
        File file = cwd;
        while (file != null) {
            File mms = new File(file, "mms");
            if (mms.exists()) {
                return file;
            }
            file = file.getParentFile();
        }
        return cwd;
    }

    private void enableAsyncLogging() {
        enableAsyncLogging(Logger.getRootLogger());
        enableAsyncLogging(Logger.getLogger(MODEL_METRICS_LOGGER));
        enableAsyncLogging(Logger.getLogger(MODEL_LOGGER));
        enableAsyncLogging(Logger.getLogger(MMS_METRICS_LOGGER));
        enableAsyncLogging(Logger.getLogger("ACCESS_LOG"));
        enableAsyncLogging(Logger.getLogger("com.amazonaws.ml.mms"));
    }

    private void enableAsyncLogging(Logger logger) {
        AsyncAppender asyncAppender = new AsyncAppender();

        @SuppressWarnings("unchecked")
        Enumeration<Appender> en = logger.getAllAppenders();
        while (en.hasMoreElements()) {
            Appender appender = en.nextElement();
            if (appender instanceof AsyncAppender) {
                // already async
                return;
            }

            logger.removeAppender(appender);
            asyncAppender.addAppender(appender);
        }
        logger.addAppender(asyncAppender);
    }

    private static String getCanonicalPath(File file) {
        try {
            return file.getCanonicalPath();
        } catch (IOException e) {
            return file.getAbsolutePath();
        }
    }

    private static String getCanonicalPath(String path) {
        if (path == null) {
            return null;
        }
        return getCanonicalPath(new File(path));
    }

    private static int getAvailableGpu() {
        try {
            Process process =
                    Runtime.getRuntime().exec("nvidia-smi --query-gpu=index --format=csv");
            int ret = process.waitFor();
            if (ret != 0) {
                return 0;
            }
            List<String> list = IOUtils.readLines(process.getInputStream(), StandardCharsets.UTF_8);
            if (list.isEmpty() || !"index".equals(list.get(0))) {
                throw new AssertionError("Unexpected nvidia-smi response.");
            }
            return list.size() - 1;
        } catch (IOException | InterruptedException e) {
            return 0;
        }
    }

    public static final class Arguments {

        private String mmsConfigFile;
        private String pythonExecutable;
        private String modelStore;
        private String[] models;

        public Arguments() {}

        public Arguments(CommandLine cmd) {
            mmsConfigFile = cmd.getOptionValue("mms-config-file");
            pythonExecutable = cmd.getOptionValue("python");
            modelStore = cmd.getOptionValue("model-store");
            models = cmd.getOptionValues("models");
        }

        public static Options getOptions() {
            Options options = new Options();
            options.addOption(
                    Option.builder("f")
                            .longOpt("mms-config-file")
                            .hasArg()
                            .argName("MMS-CONFIG-FILE")
                            .desc("Path to the configuration properties file.")
                            .build());
            options.addOption(
                    Option.builder("e")
                            .longOpt("python")
                            .hasArg()
                            .argName("PYTHON")
                            .desc("Python runtime executable path.")
                            .build());
            options.addOption(
                    Option.builder("m")
                            .longOpt("models")
                            .hasArgs()
                            .argName("MODELS")
                            .desc("Models to be loaded at startup.")
                            .build());
            options.addOption(
                    Option.builder("s")
                            .longOpt("model-store")
                            .hasArg()
                            .argName("MODELS-STORE")
                            .desc("Model store location where models can be loaded.")
                            .build());
            return options;
        }

        public String getMmsConfigFile() {
            return mmsConfigFile;
        }

        public String getPythonExecutable() {
            return pythonExecutable;
        }

        public void setMmsConfigFile(String mmsConfigFile) {
            this.mmsConfigFile = mmsConfigFile;
        }

        public String getModelStore() {
            return modelStore;
        }

        public void setModelStore(String modelStore) {
            this.modelStore = modelStore;
        }

        public String[] getModels() {
            return models;
        }

        public void setModels(String[] models) {
            this.models = models;
        }
    }
}
