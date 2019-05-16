/*
 * Copyright 2019 GridGain Systems, Inc. and Contributors.
 *
 * Licensed under the GridGain Community Edition License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.gridgain.com/products/software/community-edition/gridgain-community-edition-license
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.gridgain.ml.python;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import org.apache.ignite.Ignite;
import org.apache.ignite.ml.IgniteModel;
import org.apache.ignite.ml.inference.storage.model.ModelStorage;
import org.apache.ignite.ml.inference.storage.model.ModelStorageFactory;

/**
 * Util class that helps to save model from Python API.
 */
public class PythonModelSaver {
    /**
     * Saves the model into file.
     *
     * @param mdl Model.
     * @param dst File destination.
     * @throws IOException If the model cannot be serialized or written.
     */
    public static void save(IgniteModel<double[], Double> mdl, String dst, Ignite ignite) throws IOException {
        byte[] serializedMdl = serialize(mdl);

        if (dst.startsWith("igfs://")) {
            ModelStorage storage = new ModelStorageFactory().getModelStorage(ignite);
            storage.putFile(dst.substring(7), serializedMdl);
        }
        else {
            File file = new File(dst);
            if (!file.exists())
                file.createNewFile();

            try (FileOutputStream fos = new FileOutputStream(file)) {
                fos.write(serializedMdl);
                fos.flush();
            }
        }
    }

    /**
     * Reads model from file.
     *
     * @param src Model source.
     * @return Model.
     * @throws IOException If model cannot be read.
     * @throws ClassNotFoundException If model cannot be deserialized.
     */
    public static IgniteModel<double[], Double> read(String src,
        Ignite ignite) throws IOException, ClassNotFoundException {
        if (src.startsWith("igfs://")) {
            ModelStorage storage = new ModelStorageFactory().getModelStorage(ignite);
            byte[] serializedMdl = storage.getFile(src.substring(7));

            return deserialize(serializedMdl);
        }
        else {
            File file = new File(src);

            if (!file.exists())
                return null;

            byte[] serializedMdl = Files.readAllBytes(file.toPath());

            return deserialize(serializedMdl);
        }
    }

    /**
     * Serialized the specified object.
     *
     * @param o Object to be serialized.
     * @return Serialized object as byte array.
     * @throws IOException In case of exception.
     */
    private static <T extends Serializable> byte[] serialize(T o) throws IOException {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(o);
            oos.flush();

            return baos.toByteArray();
        }
    }

    /**
     * Deserialized object represented as a byte array.
     *
     * @param o Serialized object.
     * @param <T> Type of serialized object.
     * @return Deserialized object.
     * @throws IOException In case of exception.
     * @throws ClassNotFoundException In case of exception.
     */
    private static <T extends Serializable> T deserialize(byte[] o) throws IOException, ClassNotFoundException {
        try (ByteArrayInputStream bais = new ByteArrayInputStream(o);
             ObjectInputStream ois = new ObjectInputStream(bais)) {

            return (T)ois.readObject();
        }
    }
}
