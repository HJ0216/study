public class Ex3_10 {
    public static void main(String[] args){
        char ch = 'A';
        char lowerCase;

        if(ch>=65 && ch<=96) {
            lowerCase = (char) (ch + 32);
            System.out.println(lowerCase);
        } else {
            System.out.println(ch);
        }

    }
}
