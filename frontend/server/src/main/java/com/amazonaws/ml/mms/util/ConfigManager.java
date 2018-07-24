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
import java.util.List;
import java.util.Properties;
import org.apache.commons.io.IOUtils;

public final class ConfigManager {

    public static final String MODEL_METRICS_LOGGER = "MODEL_METRICS";
    public static final String MODEL_LOGGER = "MODEL_LOG";
    public static final String MMS_METRICS_LOGGER = "MMS_METRICS";

    private static final String DEBUG = "debug";
    private static final String PORT = "port";
    private static final String MODEL_SERVER_HOME = "model_server_home";
    private static final String MODEL_STORE = "model_store";
    private static final String LOAD_MODELS = "load_models";
    private static final String NUMBER_OF_NETTY_THREADS = "number_of_netty_threads";
    private static final String MAX_WORKERS = "max_workers";
    private static final String JOB_QUEUE_SIZE = "job_queue_size";
    private static final String NUMBER_OF_GPU = "number_of_gpu";
    private static final String METRIC_TIME_INTERVAL = "metric_time_interval";

    private static final String USE_SSL = "use_ssl";
    private static final String KEYSTORE = "keystore";
    private static final String KEYSTORE_PASS = "keystore_pass";
    private static final String KEYSTORE_TYPE = "keystore_type";
    private static final String CERTIFICATE_FILE = "certificate_file";
    private static final String PRIVATE_KEY_FILE = "private_key_file";

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
    }

    public boolean isDebug() {
        return Boolean.getBoolean("DEBUG")
                || Boolean.parseBoolean(prop.getProperty(DEBUG, "false"));
    }

    public int getPort() {
        return getIntProperty(PORT, isUseSsl() ? 8443 : 8080);
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

    public int getNumberOfGpu() {
        return getIntProperty(NUMBER_OF_GPU, 0);
    }

    public int getMetricTimeInterval() {
        return getIntProperty(METRIC_TIME_INTERVAL, 60);
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
        return getProperty(LOAD_MODELS, "NONE");
    }

    public boolean isUseSsl() {
        return Boolean.parseBoolean(prop.getProperty(USE_SSL, "false"));
    }

    public SslContext getSslContext() throws IOException, GeneralSecurityException {
        if (!isUseSsl()) {
            return null;
        }

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
}
