/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitudf;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.TypeInsnNode;
import org.objectweb.asm.tree.VarInsnNode;
import org.objectweb.asm.tree.analysis.Analyzer;
import org.objectweb.asm.tree.analysis.AnalyzerException;
import org.objectweb.asm.tree.analysis.Frame;
import org.objectweb.asm.tree.analysis.SourceInterpreter;
import org.objectweb.asm.tree.analysis.SourceValue;

/**
 * Proves, at each {@code checkcast} or {@code instanceof}, what box the value really is.
 *
 * <p>
 * Lowering to primitives erases boxes, and with them every cast between boxes. That is only right
 * when the cast cannot fail. {@code ((Boolean) (Object) x).booleanValue()} on an {@code Integer x}
 * throws {@code ClassCastException} in Java; erased, it becomes "x != 0" and the query gets a
 * number instead of an error. {@code Integer} and {@code Boolean} are both an {@code int} on the
 * JVM stack, so nothing downstream notices.
 * </p>
 *
 * <p>
 * So the analysis tracks where each reference came from -- a plain dataflow over the method, using
 * ASM's source-value frames -- and assigns it the box it must be at run time. In this method's
 * world there are only a few producers of references, and each has a known runtime class:
 * </p>
 *
 * <ul>
 * <li>a parameter: its declared type, which the analyzer has already matched to the column, and
 *     which no instruction can overwrite, because {@code astore} is refused;</li>
 * <li>{@code Integer.valueOf} and the other boxing calls: the box they return;</li>
 * <li>{@code Seq.apply} on a LIST argument: the box of the column's element type, which is what
 *     Spark's converter puts in the sequence;</li>
 * <li>an earlier, already proven {@code checkcast}: its target;</li>
 * <li>{@code aconst_null}: nothing, which passes every cast.</li>
 * </ul>
 *
 * <p>
 * A cast is erased only when every producer reaching it is proven to be exactly the target box (or
 * null). Every box this integration carries is a {@code final} class, so "exactly" is the whole
 * subtype relation. A producer proven to be a different box would throw in Java; one whose box
 * cannot be established might. Both are refused -- the precise reason is reported -- and every other
 * cast in every other UDF is unaffected.
 * </p>
 */
final class CastProof {

    private final MethodNode method;
    private final Frame<SourceValue>[] frames;
    private final Map<AbstractInsnNode, Integer> indexOf = new java.util.IdentityHashMap<>();
    private final Map<Integer, String> parameterTypes = new HashMap<>();
    private final ArrayParameters arrays;
    private final Map<Integer, InputSpec> listSlots = new HashMap<>();

    private CastProof(MethodNode method, Frame<SourceValue>[] frames, ArrayParameters arrays, List<InputSpec> inputs) {
        this.method = method;
        this.frames = frames;
        this.arrays = arrays;
        AbstractInsnNode[] insns = method.instructions.toArray();
        for (int i = 0; i < insns.length; i++) {
            indexOf.put(insns[i], i);
        }
        int slot = (method.access & Opcodes.ACC_STATIC) != 0 ? 0 : 1;
        Type[] arguments = Type.getArgumentTypes(method.desc);
        for (int i = 0; i < arguments.length; i++) {
            if (arguments[i].getSort() == Type.OBJECT) {
                parameterTypes.put(slot, arguments[i].getInternalName());
            }
            if (inputs != null && i < inputs.size() && inputs.get(i).isList()) {
                listSlots.put(slot, inputs.get(i));
            }
            slot += arguments[i].getSize();
        }
    }

    /** Builds the proof for {@code method}; {@code inputs} may be null for a helper. */
    static CastProof of(String owner, MethodNode method, ArrayParameters arrays, List<InputSpec> inputs) {
        Frame<SourceValue>[] frames;
        try {
            frames = new Analyzer<>(new SourceInterpreter()).analyze(owner, method);
        } catch (AnalyzerException e) {
            frames = null;
        }
        return new CastProof(method, frames, arrays, inputs);
    }

    /** Why the cast or type test at {@code insn} cannot be erased, or null when it provably cannot fail. */
    String whyUnproven(TypeInsnNode insn) {
        if (frames == null) {
            return "the method's dataflow could not be analysed, so the cast to " + insn.desc.replace('/', '.') + " cannot be proven";
        }
        Integer at = indexOf.get(insn);
        if (at == null || frames[at] == null) {
            return "the cast to " + insn.desc.replace('/', '.') + " is not reachable in the analysed control flow";
        }
        Frame<SourceValue> frame = frames[at];
        SourceValue operand = frame.getStack(frame.getStackSize() - 1);
        for (AbstractInsnNode producer : operand.insns) {
            String box = runtimeBoxOf(producer, 0);
            if (box == null) {
                return "the value cast to " + insn.desc.replace('/', '.') + " has a runtime type this analysis cannot establish, so the cast might throw ClassCastException";
            }
            if (box.equals(NULL)) {
                continue;
            }
            if (!box.equals(insn.desc)) {
                return "the UDF casts a " + box.replace('/', '.') + " to " + insn.desc.replace('/', '.')
                        + ", which throws ClassCastException in Java; erasing the cast would turn the exception into a number";
            }
        }
        return null;
    }

    private static final String NULL = "<null>";

    /** The box {@code producer} pushes at run time, {@link #NULL} for a null, or null when unknown. */
    private String runtimeBoxOf(AbstractInsnNode producer, int depth) {
        if (depth > 8) {
            return null;
        }
        int op = producer.getOpcode();
        if (op == Opcodes.ACONST_NULL) {
            return NULL;
        }
        if (op == Opcodes.ALOAD && producer instanceof VarInsnNode load) {
            if (listSlots.containsKey(load.var)) {
                return null; // the list itself, not a box
            }
            return parameterTypes.get(load.var);
        }
        if (op == Opcodes.CHECKCAST && producer instanceof TypeInsnNode cast) {
            // Only a cast that is itself proven passes its type on; whyUnproven is called on every
            // cast in the method, so an unproven one has already refused the UDF.
            return BoxConversions.isSupportedBox(cast.desc) ? cast.desc : null;
        }
        if (producer instanceof MethodInsnNode call) {
            if (op == Opcodes.INVOKESTATIC && BoxConversions.isSupportedBox(call.owner) && call.name.equals("valueOf") && BoxConversions.isKnown(call)) {
                return Type.getReturnType(call.desc).getInternalName();
            }
            if (op == Opcodes.INVOKEINTERFACE && call.name.equals("apply") && arrays.readsAListParameter(producer)) {
                InputSpec list = listSlots.get(arrays.slotFor(producer));
                return list == null ? null : list.kind().boxInternalName();
            }
        }
        return null;
    }
}
