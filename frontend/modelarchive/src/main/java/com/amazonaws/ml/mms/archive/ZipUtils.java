package com.amazonaws.ml.mms.archive;

import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

public final class ZipUtils {

    private ZipUtils() {}

    public static void zip(File src, File dest, boolean includeRootDir) throws IOException {
        int prefix = src.getCanonicalPath().length();
        if (includeRootDir) {
            prefix -= src.getName().length();
        }
        try (ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(dest))) {
            addToZip(prefix, src, null, zos);
        }
    }

    public static void unzip(File src, File dest) throws IOException {
        try (ZipInputStream zis = new ZipInputStream(new FileInputStream(src))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                String name = entry.getName();
                File file = new File(dest, name);
                if (entry.isDirectory()) {
                    if (!file.exists() && !file.mkdirs()) {
                        throw new IOException("Failed to create directory: " + name);
                    }
                } else {
                    try (OutputStream os = new FileOutputStream(file)) {
                        copy(zis, os);
                    }
                }
            }
        }
    }

    public static void copy(InputStream is, OutputStream os) throws IOException {
        byte[] buf = new byte[8192];
        int read;
        while ((read = is.read(buf)) != -1) {
            os.write(buf, 0, read);
        }
    }

    public static void addToZip(int prefix, File file, FileFilter filter, ZipOutputStream zos)
            throws IOException {
        String name = file.getCanonicalPath().substring(prefix);
        if (file.isDirectory()) {
            if (!name.isEmpty()) {
                ZipEntry entry = new ZipEntry(name + '/');
                zos.putNextEntry(entry);
            }
            File[] files = file.listFiles(filter);
            if (files != null) {
                for (File f : files) {
                    addToZip(prefix, f, filter, zos);
                }
            }
        } else if (file.isFile()) {
            ZipEntry entry = new ZipEntry(name);
            zos.putNextEntry(entry);
            try (FileInputStream fis = new FileInputStream(file)) {
                copy(fis, zos);
            }
        }
    }
}
