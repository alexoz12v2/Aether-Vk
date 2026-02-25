// File scope namespace (From C# 10.0)
namespace App1;

internal sealed class MainWindowViewModel
{
    public MainWindowViewModel(string MyText)
    {
        this.MyText = MyText;
    }

    public string MyText { get; set; }
}