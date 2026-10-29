using System.Collections.Generic;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

internal delegate bool AddParticleSystemReturns(ParticleSystemModel model, ParticleSystemJet jet, out ulong id);

/// <summary>
/// A jet the core refuses is never silent (reviewer report 2026-10-29: of four jets saved in the
/// model only three were in the simulation, and nothing said so): it carries
/// <see cref="JetViewModel.NativeError"/> until a later attempt succeeds.
/// </summary>
public class JetNativeSyncTests
{
  private static (Mock<INativeRuntimeService> runtime, ModelSession session, Queue<(bool ok, ulong id)> answers) Rig(int jets)
  {
    var session = new ModelSession();
    for (int i = 0; i < jets; i++)
      session.Jets.Add(new JetViewModel(session.DefaultGrain));
    var runtime = new Mock<INativeRuntimeService>();
    ulong firstId = 1;
    runtime.Setup(r => r.AddFirstParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out firstId))
      .Returns(new ParticleSystemComputedProperties(0.05f, 1f));
    var answers = new Queue<(bool ok, ulong id)>();
    runtime.Setup(r => r.AddParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out It.Ref<ulong>.IsAny))
      .Returns(new AddParticleSystemReturns((ParticleSystemModel _, ParticleSystemJet _, out ulong id) =>
      {
        var (ok, next) = answers.Dequeue();
        id = next;
        return ok;
      }));
    return (runtime, session, answers);
  }

  [Fact]
  public void ARefusedJet_IsMarked_AndTheNextEnsureRetriesOnlyThatJet()
  {
    var (runtime, session, answers) = Rig(jets: 4);
    answers.Enqueue((false, 0)); // the second jet: refused
    answers.Enqueue((true, 3));
    answers.Enqueue((true, 4));

    int refused = JetNativeSync.EnsureNative(runtime.Object, session, 1.7f);

    Assert.Equal(1, refused);
    var j = session.Jets;
    Assert.Equal(new ulong[] { 1, 0, 3, 4 }, new[] { j[0].NativePsId, j[1].NativePsId, j[2].NativePsId, j[3].NativePsId });
    Assert.Null(j[0].NativeError);
    Assert.Equal(JetNativeSync.RefusedText, j[1].NativeError);
    Assert.Null(j[2].NativeError);
    Assert.Null(j[3].NativeError);

    // the retry asks for the refused jet only, as a later system (the scene already has one)
    answers.Enqueue((true, 5));
    Assert.Equal(0, JetNativeSync.EnsureNative(runtime.Object, session, 1.7f));
    Assert.Equal(5ul, j[1].NativePsId);
    Assert.Null(j[1].NativeError);
    runtime.Verify(r => r.AddFirstParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out It.Ref<ulong>.IsAny), Times.Once);
    runtime.Verify(r => r.AddParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out It.Ref<ulong>.IsAny), Times.Exactly(4));
  }

  [Fact]
  public void AnIdTheCallDidNotStandBehind_IsNotKept()
  {
    var (runtime, session, answers) = Rig(jets: 2);
    answers.Enqueue((false, 42)); // false with a stale id: the jet is still not native

    JetNativeSync.EnsureNative(runtime.Object, session, 1.7f);

    Assert.Equal(0ul, session.Jets[1].NativePsId);
    Assert.NotNull(session.Jets[1].NativeError);
  }

  [Fact]
  public void ARefusedFirstJet_LeavesTheNextOneAsTheFirstSystem()
  {
    var (runtime, session, _) = Rig(jets: 2);
    ulong none = 0;
    runtime.Setup(r => r.AddFirstParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out none))
      .Returns((ParticleSystemComputedProperties?)null);

    Assert.Equal(2, JetNativeSync.EnsureNative(runtime.Object, session, 1.7f));

    Assert.All(session.Jets, j => Assert.Equal(JetNativeSync.RefusedText, j.NativeError));
    // neither went through AddParticleSystem: with no system in the scene each is a "first"
    runtime.Verify(r => r.AddParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out It.Ref<ulong>.IsAny), Times.Never);
  }
}
