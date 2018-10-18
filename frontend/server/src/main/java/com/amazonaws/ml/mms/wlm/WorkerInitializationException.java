package com.amazonaws.ml.mms.wlm;

public class WorkerInitializationException extends Exception {

    static final long serialVersionUID = 1L;

    /** Creates a new {@code WorkerInitializationException} instance. */
    public WorkerInitializationException() {
        super();
    }

    /**
     * Constructs a new {@code WorkerInitializationException} with the specified detail message and
     * cause.
     *
     * @param message the detail message (which is saved for later retrieval by the {@link
     *     #getMessage()} method).
     */
    public WorkerInitializationException(String message) {
        super(message);
    }

    /**
     * Constructs a new {@code WorkerInitializationException} with the specified detail message and
     * cause.
     *
     * @param message the detail message (which is saved for later retrieval by the {@link
     *     #getMessage()} method).
     * @param cause the cause (which is saved for later retrieval by the {@link #getCause()}
     *     method). (A <tt>null</tt> value is permitted, and indicates that the cause is nonexistent
     *     or unknown.)
     */
    public WorkerInitializationException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * Constructs a new {@code WorkerInitializationException} with the specified detail message and
     * cause.
     *
     * @param cause the cause (which is saved for later retrieval by the {@link #getCause()}
     *     method). (A <tt>null</tt> value is permitted, and indicates that the cause is nonexistent
     *     or unknown.)
     */
    public WorkerInitializationException(Throwable cause) {
        super(cause);
    }
}
